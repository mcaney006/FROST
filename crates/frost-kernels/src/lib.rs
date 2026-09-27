//! Mojo sampling kernels (top-k, softmax + top-p) behind a C ABI, plus a scalar Rust
//! [`reference`] with identical semantics.
//!
//! build.rs compiles `native/kernels/frost_sampling.mojo` with the *native* Mojo driver
//! (no Python anywhere) into `$OUT_DIR/libfrost_kernels.dylib`. At runtime the dylib is
//! `dlopen`ed from, in order: `$FROST_KERNELS_DYLIB`, the build-time path, `<exe dir>/`,
//! `<exe dir>/../Frameworks/` (FROST.app). The dylib itself needs the four Mojo runtime
//! dylibs (`libKGENCompilerRTShared`, `libAsyncRTMojoBindings`, `libMSupportGlobals`,
//! `libAsyncRTRuntimeGlobals`) via its rpath (`@loader_path` or the SDK lib dir).
//! If anything is missing every entry point silently uses [`reference`]; check [`backend`].
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::OnceLock;

/// C-ABI contract version compiled into the dylib (`frost_kernels_abi_version`).
pub const ABI_VERSION: i32 = 1;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum KernelError {
    #[error("empty input")]
    EmptyInput,
    #[error("k must be > 0")]
    ZeroK,
    #[error("k ({k}) exceeds n ({n})")]
    KTooLarge { k: usize, n: usize },
    #[error("NaN in logits")]
    NaN,
    #[error("temperature must be finite and > 0, got {0}")]
    InvalidTemperature(f32),
    #[error("top_p must be in (0, 1], got {0}")]
    InvalidTopP(f32),
    #[error("degenerate distribution: max logit is not finite")]
    Degenerate,
    #[error("kernel returned unknown code {0}")]
    Backend(i32),
}

/// `"mojo-dylib"` when the compiled kernel is loaded, else `"unavailable"`.
pub fn backend() -> &'static str {
    if kernels().is_some() { "mojo-dylib" } else { "unavailable" }
}

/// Why [`backend`] is `"unavailable"`, if it is.
pub fn load_error() -> Option<&'static str> {
    LOADED.get_or_init(load).as_ref().err().map(String::as_str)
}

/// Top-k logits, descending, ties broken by lower index. `-inf` is allowed.
pub fn topk(logits: &[f32], k: usize) -> Result<Vec<(u32, f32)>, KernelError> {
    let Some(kr) = kernels() else { return reference::topk(logits, k) };
    let n = logits.len();
    let m = k.min(n); // kernel rejects k > n before writing anything
    let (mut idx, mut val) = (vec![0u32; m], vec![0f32; m]);
    let code = unsafe {
        (kr.topk)(logits.as_ptr() as usize, n as isize, k as isize, idx.as_mut_ptr() as usize, val.as_mut_ptr() as usize)
    };
    check(code, n, k, 1.0, 1.0)?;
    Ok(idx.into_iter().zip(val).collect())
}

/// Temperature-scaled stable softmax, sorted descending, truncated to the smallest prefix
/// whose mass reaches `top_p`, renormalized. Returns `(index, probability)`.
pub fn softmax_topp(logits: &[f32], temperature: f32, top_p: f32) -> Result<Vec<(u32, f32)>, KernelError> {
    let Some(kr) = kernels() else { return reference::softmax_topp(logits, temperature, top_p) };
    let n = logits.len();
    let (mut prob, mut idx, mut count) = (vec![0f32; n], vec![0u32; n], 0isize);
    let code = unsafe {
        (kr.softmax)(
            logits.as_ptr() as usize,
            n as isize,
            temperature,
            top_p,
            prob.as_mut_ptr() as usize,
            idx.as_mut_ptr() as usize,
            (&mut count as *mut isize) as usize,
        )
    };
    check(code, n, 0, temperature, top_p)?;
    let count = usize::try_from(count).unwrap_or(0).min(n);
    Ok(idx[..count].iter().copied().zip(prob[..count].iter().copied()).collect())
}

fn check(code: i32, n: usize, k: usize, temperature: f32, top_p: f32) -> Result<(), KernelError> {
    Err(match code {
        0 => return Ok(()),
        -1 => KernelError::EmptyInput,
        -2 => KernelError::ZeroK,
        -3 => KernelError::KTooLarge { k, n },
        -4 => KernelError::NaN,
        -5 => KernelError::InvalidTemperature(temperature),
        -6 => KernelError::InvalidTopP(top_p),
        -7 => KernelError::Degenerate,
        c => KernelError::Backend(c),
    })
}

/// Scalar Rust implementations; the Mojo kernels must agree with these exactly (indices,
/// errors, ordering) and to ~1e-6 relative on probabilities (f64 `exp` may differ by an ulp).
pub mod reference {
    use super::KernelError;

    pub fn topk(logits: &[f32], k: usize) -> Result<Vec<(u32, f32)>, KernelError> {
        let n = logits.len();
        if n == 0 {
            return Err(KernelError::EmptyInput);
        }
        if k == 0 {
            return Err(KernelError::ZeroK);
        }
        if k > n {
            return Err(KernelError::KTooLarge { k, n });
        }
        if logits.iter().any(|v| v.is_nan()) {
            return Err(KernelError::NaN);
        }
        let mut order: Vec<u32> = (0..n as u32).collect();
        // no NaN left, so partial_cmp is total
        order.sort_by(|&a, &b| logits[b as usize].partial_cmp(&logits[a as usize]).unwrap().then(a.cmp(&b)));
        order.truncate(k);
        Ok(order.into_iter().map(|i| (i, logits[i as usize])).collect())
    }

    pub fn softmax_topp(logits: &[f32], temperature: f32, top_p: f32) -> Result<Vec<(u32, f32)>, KernelError> {
        let n = logits.len();
        if n == 0 {
            return Err(KernelError::EmptyInput);
        }
        if temperature.is_nan() || temperature <= 0.0 {
            return Err(KernelError::InvalidTemperature(temperature));
        }
        if top_p.is_nan() || top_p <= 0.0 || top_p > 1.0 {
            return Err(KernelError::InvalidTopP(top_p));
        }
        if logits.iter().any(|v| v.is_nan()) {
            return Err(KernelError::NaN);
        }
        // Same operation order as the kernel so rounding matches: z = v * (1/t), max, exp, sum.
        let inv_t = 1.0f64 / temperature as f64;
        let z: Vec<f64> = logits.iter().map(|&v| v as f64 * inv_t).collect();
        let mx = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !mx.is_finite() {
            return Err(KernelError::Degenerate);
        }
        let mut p: Vec<f64> = z.iter().map(|z| (z - mx).exp()).collect();
        let total: f64 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= total);
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| p[b as usize].partial_cmp(&p[a as usize]).unwrap().then(a.cmp(&b)));
        let mut count = n;
        let mut cum = 0.0f64;
        for (i, &j) in order.iter().enumerate() {
            cum += p[j as usize];
            if cum >= top_p as f64 {
                count = i + 1;
                break;
            }
        }
        let kept: f64 = order[..count].iter().map(|&j| p[j as usize]).sum();
        Ok(order[..count].iter().map(|&j| (j, (p[j as usize] / kept) as f32)).collect())
    }
}

// ---- dylib loading (std-only FFI to the system dynamic loader) ----------------------------

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;

type AbiVersionFn = unsafe extern "C" fn() -> i32;
type TopkFn = unsafe extern "C" fn(usize, isize, isize, usize, usize) -> i32;
type SoftmaxFn = unsafe extern "C" fn(usize, isize, f32, f32, usize, usize, usize) -> i32;

struct Kernels {
    topk: TopkFn,
    softmax: SoftmaxFn,
}

static LOADED: OnceLock<Result<Kernels, String>> = OnceLock::new();

fn kernels() -> Option<&'static Kernels> {
    LOADED.get_or_init(load).as_ref().ok()
}

fn load() -> Result<Kernels, String> {
    if cfg!(mojo_unavailable) {
        return Err("dylib was not built: Mojo compiler unavailable at build time (see cargo warnings)".into());
    }
    let mut errs = Vec::new();
    for path in candidates() {
        match try_load(&path) {
            Ok(k) => return Ok(k),
            Err(e) => errs.push(format!("{}: {e}", path.display())),
        }
    }
    Err(errs.join("; "))
}

fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("FROST_KERNELS_DYLIB") {
        v.push(PathBuf::from(p));
    }
    v.push(PathBuf::from(env!("FROST_KERNELS_DYLIB")));
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from)) {
        v.push(dir.join("libfrost_kernels.dylib"));
        v.push(dir.join("../Frameworks/libfrost_kernels.dylib"));
    }
    v
}

fn try_load(path: &std::path::Path) -> Result<Kernels, String> {
    let cpath = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
    let handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        return Err(last_dl_error());
    }
    let sym = |name: &str| -> Result<*mut c_void, String> {
        let c = CString::new(name).unwrap();
        let p = unsafe { dlsym(handle, c.as_ptr()) };
        if p.is_null() { Err(format!("missing symbol {name}: {}", last_dl_error())) } else { Ok(p) }
    };
    // SAFETY: symbol types match the `abi("C")` signatures in frost_sampling.mojo; the version
    // check guards against loading a dylib built for a different contract.
    unsafe {
        let version: AbiVersionFn = std::mem::transmute(sym("frost_kernels_abi_version")?);
        let got = version();
        if got != ABI_VERSION {
            return Err(format!("abi version {got}, expected {ABI_VERSION}"));
        }
        Ok(Kernels {
            topk: std::mem::transmute::<*mut c_void, TopkFn>(sym("frost_topk_f32")?),
            softmax: std::mem::transmute::<*mut c_void, SoftmaxFn>(sym("frost_softmax_topp_f32")?),
        })
    }
}

fn last_dl_error() -> String {
    let p = unsafe { dlerror() };
    if p.is_null() { "unknown dlopen error".into() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FROST_REQUIRE_MOJO=1 turns "reference fallback" into a hard failure; never a skip.
    fn require_mojo_if_asked() {
        if std::env::var("FROST_REQUIRE_MOJO").as_deref() == Ok("1") {
            assert_eq!(backend(), "mojo-dylib", "Mojo backend required but unavailable: {:?}", load_error());
        }
    }

    // xorshift64*: deterministic random logits in [-8, 8) without a dependency.
    fn random_logits(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
                (r as f32 / (1u64 << 24) as f32) * 16.0 - 8.0
            })
            .collect()
    }

    fn assert_prob_parity(got: &[(u32, f32)], want: &[(u32, f32)]) {
        assert_eq!(got.len(), want.len(), "count differs");
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.0, w.0, "index differs");
            assert!((g.1 - w.1).abs() <= 1e-6 * w.1.abs().max(1e-30) + 1e-9, "prob {} vs {}", g.1, w.1);
        }
    }

    #[test]
    fn backend_reports_and_honours_requirement() {
        require_mojo_if_asked();
        assert!(matches!(backend(), "mojo-dylib" | "unavailable"));
        if backend() == "unavailable" {
            eprintln!("frost-kernels: reference fallback active: {:?}", load_error());
        }
    }

    #[test]
    fn topk_parity_with_reference() {
        require_mojo_if_asked();
        for (i, &n) in [1usize, 7, 64, 131_072].iter().enumerate() {
            let logits = random_logits(n, 0xC0FFEE + i as u64);
            for k in [1usize, 5, 50] {
                assert_eq!(topk(&logits, k), reference::topk(&logits, k), "n={n} k={k}");
            }
        }
    }

    #[test]
    fn softmax_topp_parity_with_reference() {
        require_mojo_if_asked();
        for (i, &n) in [1usize, 7, 64, 131_072].iter().enumerate() {
            let logits = random_logits(n, 0xBEEF + i as u64);
            for (t, p) in [(1.0f32, 1.0f32), (0.7, 0.1), (2.0, 0.5)] {
                let got = softmax_topp(&logits, t, p).unwrap();
                let want = reference::softmax_topp(&logits, t, p).unwrap();
                assert_prob_parity(&got, &want);
                let mass: f64 = got.iter().map(|x| x.1 as f64).sum();
                assert!((mass - 1.0).abs() < 1e-6, "renormalized mass {mass} (n={n} t={t} p={p})");
                // f64 cumulative rounding may reach 1.0 a few near-zero entries early on huge n
                if p == 1.0 && n <= 64 {
                    assert_eq!(got.len(), n);
                }
            }
        }
    }

    #[test]
    fn topk_heavy_ties_and_k_up_to_n() {
        require_mojo_if_asked();
        // many duplicate values: eviction must keep the lower index among equals
        let ties: Vec<f32> = (0..1000).map(|i| (i % 5) as f32).collect();
        for k in [1usize, 37, 200, 999, 1000] {
            assert_eq!(topk(&ties, k), reference::topk(&ties, k), "ties k={k}");
        }
        let logits = random_logits(64, 99);
        for k in [63usize, 64] {
            assert_eq!(topk(&logits, k), reference::topk(&logits, k), "k={k}");
        }
        assert_eq!(topk(&[f32::NEG_INFINITY; 4], 4).unwrap().iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn topk_edge_cases() {
        require_mojo_if_asked();
        assert_eq!(topk(&[], 1), Err(KernelError::EmptyInput));
        assert_eq!(topk(&[1.0], 0), Err(KernelError::ZeroK));
        assert_eq!(topk(&[1.0, 2.0], 3), Err(KernelError::KTooLarge { k: 3, n: 2 }));
        assert_eq!(topk(&[1.0, f32::NAN], 1), Err(KernelError::NaN));
        let v = [1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 3.0];
        assert_eq!(topk(&v, 5).unwrap(), vec![(1, f32::INFINITY), (2, 3.0), (4, 3.0), (0, 1.0), (3, f32::NEG_INFINITY)]);
        assert_eq!(topk(&v, 5), reference::topk(&v, 5));
    }

    #[test]
    fn softmax_topp_edge_cases() {
        require_mojo_if_asked();
        assert_eq!(softmax_topp(&[], 1.0, 1.0), Err(KernelError::EmptyInput));
        assert_eq!(softmax_topp(&[1.0], 0.0, 1.0), Err(KernelError::InvalidTemperature(0.0)));
        assert_eq!(softmax_topp(&[1.0], -1.0, 1.0), Err(KernelError::InvalidTemperature(-1.0)));
        assert_eq!(softmax_topp(&[1.0], 1.0, 0.0), Err(KernelError::InvalidTopP(0.0)));
        assert_eq!(softmax_topp(&[1.0], 1.0, 1.5), Err(KernelError::InvalidTopP(1.5)));
        assert_eq!(softmax_topp(&[1.0, f32::NAN], 1.0, 1.0), Err(KernelError::NaN));
        assert_eq!(softmax_topp(&[1.0, f32::INFINITY], 1.0, 1.0), Err(KernelError::Degenerate));
        assert_eq!(softmax_topp(&[f32::NEG_INFINITY; 3], 1.0, 1.0), Err(KernelError::Degenerate));
        // -inf gets exactly zero mass: cum reaches 1.0 without it, so top_p=1.0 drops it
        let v = [0.0, f32::NEG_INFINITY, 1.0];
        let got = softmax_topp(&v, 1.0, 1.0).unwrap();
        assert_eq!(got.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2, 0]);
        assert_eq!(got, reference::softmax_topp(&v, 1.0, 1.0).unwrap());
        // top_p = 0.1 on a peaked distribution keeps exactly the smallest sufficient prefix
        let got = softmax_topp(&[10.0, 0.0, 0.0, 0.0], 1.0, 0.1).unwrap();
        assert_eq!(got, vec![(0, 1.0)]);
        let logits = random_logits(64, 7);
        let got = softmax_topp(&logits, 1.0, 0.1).unwrap();
        let full = reference::softmax_topp(&logits, 1.0, 1.0).unwrap();
        let cum: Vec<f32> = full.iter().scan(0.0, |c, x| { *c += x.1; Some(*c) }).collect();
        assert!(cum[got.len() - 1] >= 0.1 - 1e-6 && (got.len() == 1 || cum[got.len() - 2] < 0.1), "prefix not minimal");
    }
}

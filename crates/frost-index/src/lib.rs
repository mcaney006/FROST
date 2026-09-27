//! frost-index: safe Rust wrapper over the Zig native data operations, with a
//! scalar Rust reference used for parity testing.
//!
//! Retrieval cascade:
//!   1) binary sketch coarse search  (Hamming distance over packed bits)
//!   2) INT8 shortlist scoring        (integer dot, caller applies the scale)
//!   3) FP32 exact rerank             (dot product)
//!
//! Checked data operations:
//!   - MLX 4-bit affine weight validation + row dequant (model loader)
//!   - FROSTIDX1 memory-mapped exact vector index (write / open / search)

use std::ffi::{c_char, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;

/// Opaque Zig `FrostIndex` handle.
#[repr(C)]
struct RawIndex {
    _opaque: [u8; 0],
}

/// Mirrors Zig `FrostQ4Stats`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Q4Stats {
    pub rows: u64,
    pub cols: u64,
    /// Non-finite scales + biases.
    pub nonfinite: u64,
    pub zero_scales: u64,
    /// Min/max over finite scales (0 if there are none).
    pub min_scale: f32,
    pub max_scale: f32,
}

extern "C" {
    fn frost_hamming(a: *const u8, b: *const u8, nbytes: usize) -> u32;
    fn frost_dot_i8(a: *const i8, b: *const i8, n: usize) -> i32;
    fn frost_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;

    fn frost_q4_validate(
        w: *const u32,
        rows: usize,
        cols_packed: usize,
        scales: *const u16,
        biases: *const u16,
        groups: usize,
        group_size: u32,
        bits: u32,
        out: *mut Q4Stats,
    ) -> i32;
    fn frost_q4_dequant_row(
        w: *const u32,
        scales: *const u16,
        biases: *const u16,
        cols_packed: usize,
        group_size: u32,
        out: *mut f32,
    ) -> i32;

    fn frost_index_write(
        path: *const c_char,
        ids: *const u64,
        vectors: *const f32,
        count: u64,
        dim: u32,
        generation: u64,
    ) -> i32;
    fn frost_index_open(path: *const c_char, out: *mut *mut RawIndex) -> i32;
    fn frost_index_close(idx: *mut RawIndex);
    fn frost_index_meta(idx: *const RawIndex, dim: *mut u32, count: *mut u64, generation: *mut u64);
    fn frost_index_search(
        idx: *const RawIndex,
        query: *const f32,
        dim: u32,
        k: u32,
        out_ids: *mut u64,
        out_scores: *mut f32,
        out_n: *mut u32,
    ) -> i32;
}

/// Zig error codes (see the table at the top of native/index/frost_index.zig),
/// plus `InvalidArgument` for Rust-side length checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum IndexError {
    #[error("unsupported quantization bits (only 4-bit is supported)")]
    UnsupportedBits,
    #[error("dimension mismatch")]
    DimMismatch,
    #[error("size arithmetic overflow")]
    Overflow,
    #[error("non-finite value (NaN/Inf)")]
    NonFinite,
    #[error("invalid argument")]
    InvalidArgument,
    #[error("I/O error")]
    Io,
    #[error("bad magic: not a FROSTIDX file")]
    BadMagic,
    #[error("bad header: unsupported version, zero dim, or nonzero pad")]
    BadHeader,
    #[error("header hash mismatch")]
    BadHash,
    #[error("file length disagrees with header (truncated or trailing bytes)")]
    BadLength,
    #[error("out of memory")]
    NoMem,
    #[error("unknown native error code {0}")]
    Unknown(i32),
}

impl IndexError {
    fn check(code: i32) -> Result<(), Self> {
        Err(match code {
            0 => return Ok(()),
            -1 => Self::UnsupportedBits,
            -2 => Self::DimMismatch,
            -3 => Self::Overflow,
            -4 => Self::NonFinite,
            -5 => Self::InvalidArgument,
            -6 => Self::Io,
            -7 => Self::BadMagic,
            -8 => Self::BadHeader,
            -9 => Self::BadHash,
            -10 => Self::BadLength,
            -11 => Self::NoMem,
            c => Self::Unknown(c),
        })
    }
}

/// Hamming distance between two equal-length packed-bit sketches.
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "sketch length mismatch");
    // SAFETY: pointers valid for `len` bytes; lengths checked equal.
    unsafe { frost_hamming(a.as_ptr(), b.as_ptr(), a.len()) }
}

/// Integer dot product of two INT8 vectors (shortlist scoring).
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    unsafe { frost_dot_i8(a.as_ptr(), b.as_ptr(), a.len()) }
}

/// FP32 dot product (exact rerank).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    unsafe { frost_dot_f32(a.as_ptr(), b.as_ptr(), a.len()) }
}

/// Validates an MLX 4-bit affine quantized weight: `w` is rows x cols_packed,
/// `scales`/`biases` are rows x groups bf16 bit patterns.
#[allow(clippy::too_many_arguments)]
pub fn q4_validate(
    w: &[u32],
    rows: usize,
    cols_packed: usize,
    scales: &[u16],
    biases: &[u16],
    groups: usize,
    group_size: u32,
    bits: u32,
) -> Result<Q4Stats, IndexError> {
    let nw = rows.checked_mul(cols_packed).ok_or(IndexError::Overflow)?;
    let ns = rows.checked_mul(groups).ok_or(IndexError::Overflow)?;
    if w.len() != nw || scales.len() != ns || biases.len() != ns {
        return Err(IndexError::InvalidArgument);
    }
    let mut st = Q4Stats::default();
    // SAFETY: slice lengths match the shape Zig reads (rows*groups scales/biases).
    IndexError::check(unsafe {
        frost_q4_validate(
            w.as_ptr(),
            rows,
            cols_packed,
            scales.as_ptr(),
            biases.as_ptr(),
            groups,
            group_size,
            bits,
            &mut st,
        )
    })?;
    Ok(st)
}

/// Dequantizes one MLX 4-bit row (`w.len() * 8` values, low nibble first).
pub fn q4_dequant_row(w: &[u32], scales: &[u16], biases: &[u16], group_size: u32) -> Result<Vec<f32>, IndexError> {
    let cols = w.len().checked_mul(8).ok_or(IndexError::Overflow)?;
    if group_size == 0 {
        return Err(IndexError::InvalidArgument);
    }
    let groups = cols.div_ceil(group_size as usize);
    if scales.len() != groups || biases.len() != groups {
        return Err(IndexError::InvalidArgument);
    }
    let mut out = vec![0f32; cols];
    // SAFETY: Zig writes exactly `cols` floats and reads cols/group_size groups
    // (it rejects non-divisible shapes before reading).
    IndexError::check(unsafe {
        frost_q4_dequant_row(w.as_ptr(), scales.as_ptr(), biases.as_ptr(), w.len(), group_size, out.as_mut_ptr())
    })?;
    Ok(out)
}

fn c_path(path: &Path) -> Result<CString, IndexError> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| IndexError::InvalidArgument)
}

/// Writes a FROSTIDX1 file via `<path>.tmp` + fsync + rename (atomic swap).
/// `vectors` is `ids.len() * dim` floats, row-major, caller-normalized.
pub fn write_index(path: impl AsRef<Path>, ids: &[u64], vectors: &[f32], dim: u32, generation: u64) -> Result<(), IndexError> {
    if ids.len().checked_mul(dim as usize) != Some(vectors.len()) {
        return Err(IndexError::InvalidArgument);
    }
    let p = c_path(path.as_ref())?;
    // SAFETY: vectors.len() == ids.len() * dim, checked above.
    IndexError::check(unsafe {
        frost_index_write(p.as_ptr(), ids.as_ptr(), vectors.as_ptr(), ids.len() as u64, dim, generation)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Meta {
    pub dim: u32,
    pub count: u64,
    pub generation: u64,
}

/// A read-only memory-mapped FROSTIDX1 index.
pub struct Index {
    raw: NonNull<RawIndex>,
}

// SAFETY: the handle owns a private read-only mapping plus a heap struct that is
// never mutated after open; meta/search only read, and close runs once in Drop.
unsafe impl Send for Index {}
unsafe impl Sync for Index {}

impl Index {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IndexError> {
        let p = c_path(path.as_ref())?;
        let mut raw = std::ptr::null_mut();
        IndexError::check(unsafe { frost_index_open(p.as_ptr(), &mut raw) })?;
        Ok(Self { raw: NonNull::new(raw).expect("frost_index_open returned OK with a null handle") })
    }

    pub fn meta(&self) -> Meta {
        let mut m = Meta { dim: 0, count: 0, generation: 0 };
        unsafe { frost_index_meta(self.raw.as_ptr(), &mut m.dim, &mut m.count, &mut m.generation) };
        m
    }

    /// Exact top-k by dot product: descending score, ties broken by lower id.
    /// Returns min(k, count) hits.
    pub fn search(&self, query: &[f32], k: u32) -> Result<Vec<(u64, f32)>, IndexError> {
        let dim = u32::try_from(query.len()).map_err(|_| IndexError::DimMismatch)?;
        let k = u64::from(k).min(self.meta().count) as u32;
        let (mut ids, mut scores, mut n) = (vec![0u64; k as usize], vec![0f32; k as usize], 0u32);
        // SAFETY: query holds `dim` floats (Zig rejects dim != index dim before
        // reading); output buffers hold k = min(k, count) entries.
        IndexError::check(unsafe {
            frost_index_search(self.raw.as_ptr(), query.as_ptr(), dim, k, ids.as_mut_ptr(), scores.as_mut_ptr(), &mut n)
        })?;
        Ok(ids.into_iter().zip(scores).take(n as usize).collect())
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe { frost_index_close(self.raw.as_ptr()) }
    }
}

/// Scalar Rust reference implementations for numeric/index parity tests.
pub mod reference {
    use super::{IndexError, Q4Stats};

    fn bf16(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

    pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
        a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
    }
    pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
    }
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    /// Exhaustive top-k: descending score, ties by lower id, NaN scores skipped.
    pub fn search(ids: &[u64], vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut all: Vec<(u64, f32)> = ids
            .iter()
            .zip(vectors.chunks_exact(dim))
            .map(|(&id, v)| (id, dot_f32(query, v)))
            .filter(|(_, s)| !s.is_nan())
            .collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    pub fn q4_validate(
        rows: usize,
        cols_packed: usize,
        scales: &[u16],
        biases: &[u16],
        groups: usize,
        group_size: u32,
        bits: u32,
    ) -> Result<Q4Stats, IndexError> {
        if bits != 4 {
            return Err(IndexError::UnsupportedBits);
        }
        if group_size == 0 {
            return Err(IndexError::InvalidArgument);
        }
        let cols = cols_packed.checked_mul(8).ok_or(IndexError::Overflow)?;
        let covered = groups.checked_mul(group_size as usize).ok_or(IndexError::Overflow)?;
        if cols != covered {
            return Err(IndexError::DimMismatch);
        }
        let n = rows.checked_mul(groups).ok_or(IndexError::Overflow)?;
        let finite = |x: u16| x & 0x7F80 != 0x7F80;
        let mut st = Q4Stats { rows: rows as u64, cols: cols as u64, ..Default::default() };
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for (&s, &b) in scales[..n].iter().zip(&biases[..n]) {
            st.nonfinite += !finite(s) as u64 + !finite(b) as u64;
            if finite(s) {
                st.zero_scales += (s & 0x7FFF == 0) as u64;
                lo = lo.min(bf16(s));
                hi = hi.max(bf16(s));
            }
        }
        if lo <= hi {
            (st.min_scale, st.max_scale) = (lo, hi);
        }
        if st.nonfinite > 0 { Err(IndexError::NonFinite) } else { Ok(st) }
    }

    pub fn q4_dequant_row(w: &[u32], scales: &[u16], biases: &[u16], group_size: usize) -> Vec<f32> {
        (0..w.len() * 8)
            .map(|j| {
                let q = (w[j / 8] >> (4 * (j % 8))) & 0xF;
                let g = j / group_size;
                bf16(scales[g]) * q as f32 + bf16(biases[g])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    #[test]
    fn zig_matches_scalar_reference_random() {
        // deterministic pseudo-random vectors, parity Zig vs scalar reference
        let mut s: u64 = 0x1234_5678;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for n in [1usize, 3, 8, 64, 257] {
            let ba: Vec<u8> = (0..n).map(|_| (next() & 0xff) as u8).collect();
            let bb: Vec<u8> = (0..n).map(|_| (next() & 0xff) as u8).collect();
            assert_eq!(hamming(&ba, &bb), reference::hamming(&ba, &bb), "hamming n={n}");

            let ia: Vec<i8> = (0..n).map(|_| next() as i8).collect();
            let ib: Vec<i8> = (0..n).map(|_| next() as i8).collect();
            assert_eq!(dot_i8(&ia, &ib), reference::dot_i8(&ia, &ib), "dot_i8 n={n}");

            let fa: Vec<f32> = (0..n).map(|_| (next() % 1000) as f32 / 1000.0).collect();
            let fb: Vec<f32> = (0..n).map(|_| (next() % 1000) as f32 / 1000.0).collect();
            let d = dot_f32(&fa, &fb);
            let r = reference::dot_f32(&fa, &fb);
            assert!((d - r).abs() < 1e-3, "dot_f32 n={n}: {d} vs {r}");
        }
    }

    #[test]
    fn hamming_edge_cases() {
        assert_eq!(hamming(&[], &[]), 0);
        assert_eq!(hamming(&[0xFF], &[0x00]), 8);
        assert_eq!(hamming(&[0xAA, 0x55], &[0x55, 0xAA]), 16);
    }

    // ------------------------------------------------------------ helpers

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut s = self.0;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            self.0 = s;
            s
        }
        /// Uniform in [-1, 1).
        fn f32(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn unit_vectors(rng: &mut Rng, count: usize, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..count * dim).map(|_| rng.f32()).collect();
        for row in v.chunks_exact_mut(dim) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            row.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("frost-index-{}-{tag}.idx", std::process::id()))
    }

    /// Checks Zig top-k against the scalar reference. SIMD vs scalar summation
    /// order differs by ~1e-7, so equal-rank ids may only differ on near-ties.
    fn assert_topk(got: &[(u64, f32)], want: &[(u64, f32)], exact: impl Fn(u64) -> f32) {
        assert_eq!(got.len(), want.len());
        assert!(got.windows(2).all(|p| p[0].1 >= p[1].1), "not descending: {got:?}");
        let mut seen = std::collections::HashSet::new();
        for (g, w) in got.iter().zip(want) {
            assert!(seen.insert(g.0), "duplicate id {}", g.0);
            assert!((g.1 - exact(g.0)).abs() < 1e-5, "score/id pairing off for {g:?}");
            assert!((g.1 - w.1).abs() < 1e-5, "rank mismatch: {g:?} vs {w:?}");
        }
    }

    fn fnv1a(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
    }

    fn open_bytes(tag: &str, bytes: &[u8]) -> Result<Index, IndexError> {
        let path = tmp_path(tag);
        std::fs::write(&path, bytes).unwrap();
        let r = Index::open(&path);
        std::fs::remove_file(&path).unwrap();
        r
    }

    fn bf(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    // ------------------------------------------------------------ index

    #[test]
    fn index_round_trip_matches_reference() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for dim in [64usize, 256, 768] {
            for count in [1usize, 17, 1000] {
                let ids: Vec<u64> = (0..count as u64).map(|i| i * 7 + 3).collect();
                let vecs = unit_vectors(&mut rng, count, dim);
                let path = tmp_path(&format!("rt-{dim}-{count}"));
                write_index(&path, &ids, &vecs, dim as u32, 42).unwrap();
                let idx = Index::open(&path).unwrap();
                assert_eq!(idx.meta(), Meta { dim: dim as u32, count: count as u64, generation: 42 });

                let query = unit_vectors(&mut rng, 1, dim);
                let exact = |id: u64| reference::dot_f32(&query, &vecs[(id as usize - 3) / 7 * dim..][..dim]);
                for k in [1usize, 5, count + 5] {
                    let got = idx.search(&query, k as u32).unwrap();
                    assert_eq!(got.len(), k.min(count), "dim={dim} count={count} k={k}");
                    assert_topk(&got, &reference::search(&ids, &vecs, dim, &query, k), exact);
                }
                drop(idx);
                std::fs::remove_file(&path).unwrap();
            }
        }
    }

    #[test]
    fn empty_index() {
        let path = tmp_path("empty");
        write_index(&path, &[], &[], 16, 7).unwrap();
        let idx = Index::open(&path).unwrap();
        assert_eq!(idx.meta(), Meta { dim: 16, count: 0, generation: 7 });
        assert!(idx.search(&[0.25; 16], 10).unwrap().is_empty());
        drop(idx);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn search_errors_and_tie_break() {
        let v = [0.6f32, 0.8, 0.0, 0.0];
        let neg = [-0.6f32, -0.8, 0.0, 0.0];
        let path = tmp_path("ties");
        write_index(&path, &[5, 3, 9, 1], &[v, v, v, neg].concat(), 4, 1).unwrap();
        let idx = Index::open(&path).unwrap();

        let ids: Vec<u64> = idx.search(&v, 3).unwrap().into_iter().map(|h| h.0).collect();
        assert_eq!(ids, [3, 5, 9], "equal scores must order by lower id");
        assert!(idx.search(&v, 0).unwrap().is_empty());
        assert_eq!(idx.search(&[f32::NAN, 0.0, 0.0, 0.0], 3), Err(IndexError::NonFinite));
        assert_eq!(idx.search(&[f32::INFINITY, 0.0, 0.0, 0.0], 3), Err(IndexError::NonFinite));
        assert_eq!(idx.search(&v[..3], 3), Err(IndexError::DimMismatch));
        drop(idx);
        std::fs::remove_file(&path).unwrap();

        assert_eq!(write_index(&path, &[1, 2], &v, 4, 0), Err(IndexError::InvalidArgument));
        assert_eq!(write_index(&path, &[], &[], 0, 0), Err(IndexError::InvalidArgument));
    }

    #[test]
    fn file_format_and_corruption() {
        let (dim, count) = (3usize, 5usize);
        let mut rng = Rng(11);
        let ids: Vec<u64> = (100..100 + count as u64).collect();
        let path = tmp_path("fmt");
        write_index(&path, &ids, &unit_vectors(&mut rng, count, dim), dim as u32, 0xABCD).unwrap();
        let good = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        // Pin the on-disk contract for non-Zig readers.
        let u32_at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        assert_eq!(&good[..8], b"FROSTIDX");
        assert_eq!((u32_at(&good, 8), u32_at(&good, 12)), (1, dim as u32));
        assert_eq!((u64_at(&good, 16), u64_at(&good, 24)), (count as u64, 0xABCD));
        assert_eq!(u64_at(&good, 40), fnv1a(&good[..40]));
        assert_eq!(u64_at(&good, 48), 100);
        let vec_off = (48 + 8 * count).next_multiple_of(64);
        assert_eq!(good.len(), vec_off + 4 * count * dim);

        let mut b = good.clone();
        b.truncate(good.len() - 4);
        assert_eq!(open_bytes("trunc", &b).err(), Some(IndexError::BadLength));
        assert_eq!(open_bytes("trunc-hdr", &good[..10]).err(), Some(IndexError::BadLength));
        let mut b = good.clone();
        b.push(0);
        assert_eq!(open_bytes("trailing", &b).err(), Some(IndexError::BadLength));

        let mut b = good.clone();
        b[0] ^= 1;
        assert_eq!(open_bytes("magic", &b).err(), Some(IndexError::BadMagic));

        let mut b = good.clone();
        b[24] ^= 1; // generation byte, hash left stale
        assert_eq!(open_bytes("hash", &b).err(), Some(IndexError::BadHash));

        let mut b = good.clone();
        b[8] = 2; // version 2 with a valid hash
        let h = fnv1a(&b[..40]);
        b[40..48].copy_from_slice(&h.to_le_bytes());
        assert_eq!(open_bytes("version", &b).err(), Some(IndexError::BadHeader));

        assert_eq!(Index::open(tmp_path("does-not-exist")).err(), Some(IndexError::Io));
        assert!(open_bytes("good", &good).is_ok());
    }

    #[test]
    fn search_latency_10k_x_768() {
        let (count, dim) = (10_000usize, 768usize);
        let mut rng = Rng(0xDEAD_BEEF);
        let ids: Vec<u64> = (0..count as u64).collect();
        let vecs = unit_vectors(&mut rng, count, dim);
        let path = tmp_path("latency");
        write_index(&path, &ids, &vecs, dim as u32, 1).unwrap();
        let idx = Index::open(&path).unwrap();
        let query = unit_vectors(&mut rng, 1, dim);

        let got = idx.search(&query, 10).unwrap(); // also faults the mapping in
        assert_topk(&got, &reference::search(&ids, &vecs, dim, &query, 10), |id| {
            reference::dot_f32(&query, &vecs[id as usize * dim..][..dim])
        });
        let mut t: Vec<Duration> = (0..20)
            .map(|_| {
                let s = Instant::now();
                assert_eq!(idx.search(&query, 10).unwrap().len(), 10);
                s.elapsed()
            })
            .collect();
        t.sort();
        eprintln!("frost-index search {count}x{dim} k=10: p50 {:?} (min {:?}, max {:?}, 20 runs)", t[10], t[0], t[19]);
        drop(idx);
        std::fs::remove_file(&path).unwrap();
    }

    // ------------------------------------------------------------ q4

    #[test]
    fn q4_validate_codes() {
        use IndexError::*;
        // 2 rows x 64 cols, group 32 -> cols_packed 8, groups 2 per row
        let w = vec![0u32; 16];
        let mut scales = vec![bf(0.5), bf(2.0), bf(0.0), bf(1.0)];
        let mut biases = vec![bf(-1.0); 4];
        let st = q4_validate(&w, 2, 8, &scales, &biases, 2, 32, 4).unwrap();
        assert_eq!(st, Q4Stats { rows: 2, cols: 64, nonfinite: 0, zero_scales: 1, min_scale: 0.0, max_scale: 2.0 });
        assert_eq!(reference::q4_validate(2, 8, &scales, &biases, 2, 32, 4), Ok(st));

        assert_eq!(q4_validate(&w, 2, 8, &scales, &biases, 2, 32, 8), Err(UnsupportedBits));
        assert_eq!(q4_validate(&w, 2, 8, &scales, &biases, 2, 64, 4), Err(DimMismatch));
        assert_eq!(q4_validate(&[], 0, usize::MAX / 4, &[], &[], 0, 32, 4), Err(Overflow));
        assert_eq!(q4_validate(&w, 2, 8, &scales, &biases, 2, 0, 4), Err(InvalidArgument));
        assert_eq!(q4_validate(&w, 2, 8, &scales[..3], &biases, 2, 32, 4), Err(InvalidArgument));

        scales[1] = 0x7FC0; // NaN scale
        assert_eq!(q4_validate(&w, 2, 8, &scales, &biases, 2, 32, 4), Err(NonFinite));
        scales[1] = bf(2.0);
        biases[3] = 0xFF80; // -Inf bias
        assert_eq!(q4_validate(&w, 2, 8, &scales, &biases, 2, 32, 4), Err(NonFinite));
        assert_eq!(reference::q4_validate(2, 8, &scales, &biases, 2, 32, 4), Err(NonFinite));
    }

    #[test]
    fn q4_dequant_hand_computed_word() {
        // Low nibble first: 0x76543210 -> q = 0..7, 0xFEDCBA98 -> q = 8..15.
        let out = q4_dequant_row(&[0x7654_3210, 0xFEDC_BA98], &[bf(2.0)], &[bf(-1.0)], 16).unwrap();
        let want: Vec<f32> = (0..16).map(|q| 2.0 * q as f32 - 1.0).collect();
        assert_eq!(out, want);
    }

    #[test]
    fn q4_dequant_matches_reference_random() {
        let mut rng = Rng(7);
        // (cols_packed, group_size): vector path (gs % 8 == 0) and scalar path.
        for (cols_packed, gs) in [(16usize, 32usize), (32, 64), (64, 128), (5, 40), (3, 12)] {
            let groups = cols_packed * 8 / gs;
            let w: Vec<u32> = (0..cols_packed).map(|_| rng.next() as u32).collect();
            let scales: Vec<u16> = (0..groups).map(|_| bf(rng.f32())).collect();
            let biases: Vec<u16> = (0..groups).map(|_| bf(rng.f32())).collect();
            assert_eq!(
                q4_dequant_row(&w, &scales, &biases, gs as u32).unwrap(),
                reference::q4_dequant_row(&w, &scales, &biases, gs),
                "cols_packed={cols_packed} gs={gs}"
            );
        }
        assert_eq!(q4_dequant_row(&[0; 3], &[0; 3], &[0; 3], 10), Err(IndexError::DimMismatch));
        assert_eq!(q4_dequant_row(&[0; 2], &[0; 1], &[0; 1], 0), Err(IndexError::InvalidArgument));
        assert_eq!(q4_dequant_row(&[0; 2], &[0; 2], &[0; 1], 16), Err(IndexError::InvalidArgument));
    }
}

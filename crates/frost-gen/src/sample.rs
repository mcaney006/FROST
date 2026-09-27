//! Logit processing and sampling on the CPU (f32 logits copied from the GPU once per token).
//!
//! Pipeline: repetition penalty (Rust) -> top-k selection -> temperature softmax + top-p
//! (compiled Mojo kernels via `frost-kernels`, Rust reference when the dylib is absent)
//! -> categorical draw (Rust). `temperature == 0` is greedy argmax in Rust. The RNG is a
//! seeded xorshift so runs are reproducible. Every call reports which backend executed.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleParams {
    pub temperature: f32,
    /// 0 disables.
    pub top_k: usize,
    /// 1.0 disables.
    pub top_p: f32,
    /// 1.0 disables; applied to the last `repeat_window` generated ids.
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        // Mistral's published recommendation for the Ministral 3 family is a low temperature.
        // top_k 64 keeps the per-token kernel path O(n) selection + a 64-wide softmax.
        SampleParams { temperature: 0.15, top_k: 64, top_p: 1.0, repeat_penalty: 1.0, repeat_window: 64, seed: 0x5eed }
    }
}

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng { Rng(seed.max(1)) }
    pub fn next_u64(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 }
}

/// Which implementation actually executed the logit processing for a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend { GreedyRust, MojoDylib, RustReference }

impl Backend {
    pub fn label(self) -> &'static str {
        match self { Backend::GreedyRust => "rust-argmax", Backend::MojoDylib => "mojo-dylib", Backend::RustReference => "rust-reference" }
    }
}

fn kernel_backend() -> Backend { if frost_kernels::backend() == "mojo-dylib" { Backend::MojoDylib } else { Backend::RustReference } }

/// Pick the next token. `logits` is consumed (modified in place).
pub fn sample(logits: &mut [f32], p: &SampleParams, recent: &[u32], rng: &mut Rng) -> (u32, Backend) {
    assert!(!logits.is_empty(), "empty logits");
    if p.repeat_penalty != 1.0 && p.repeat_penalty > 0.0 {
        let start = recent.len().saturating_sub(p.repeat_window);
        for &t in &recent[start..] {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 { *l / p.repeat_penalty } else { *l * p.repeat_penalty };
            }
        }
    }
    if p.temperature <= 0.0 { return (argmax(logits), Backend::GreedyRust); }

    let n = logits.len();
    let backend = kernel_backend();
    // 1) candidate selection (top-k) — compiled kernel; identity when top_k is off
    let (cand_idx, cand_val): (Vec<u32>, Vec<f32>) = if p.top_k > 0 && p.top_k < n {
        match frost_kernels::topk(logits, p.top_k) {
            Ok(v) => v.into_iter().unzip(),
            Err(_) => return (argmax(logits), Backend::GreedyRust), // NaN/degenerate logits: never sample garbage
        }
    } else {
        ((0..n as u32).collect(), logits.to_vec())
    };
    // 2) temperature softmax + top-p over the candidates — compiled kernel
    let top_p = if p.top_p <= 0.0 || p.top_p > 1.0 { 1.0 } else { p.top_p };
    let dist = match frost_kernels::softmax_topp(&cand_val, p.temperature, top_p) {
        Ok(d) if !d.is_empty() => d,
        _ => return (argmax(logits), Backend::GreedyRust), // NaN/degenerate: never sample garbage
    };
    // 3) categorical draw over the renormalized prefix
    let mut r = rng.next_f32();
    for &(li, prob) in &dist {
        r -= prob;
        if r <= 0.0 { return (cand_idx[li as usize], backend); }
    }
    (cand_idx[dist[dist.len() - 1].0 as usize], backend)
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] || logits[best].is_nan() { best = i; }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_nan_safety() {
        let mut l = vec![0.1, f32::NAN, 3.0, 2.0];
        let p = SampleParams { temperature: 0.0, ..Default::default() };
        assert_eq!(sample(&mut l, &p, &[], &mut Rng::new(1)).0, 2);
        let mut l2 = vec![f32::NAN, 1.0];
        assert_eq!(argmax(&l2), 1);
        let p2 = SampleParams { temperature: 1.0, top_k: 0, ..Default::default() };
        assert_eq!(sample(&mut l2, &p2, &[], &mut Rng::new(1)).0, 1, "NaN must never be sampled");
    }

    #[test]
    fn top_k_and_top_p_restrict_support() {
        let mut rng = Rng::new(42);
        let base = vec![10.0, 9.0, 1.0, 0.0, -5.0];
        let p = SampleParams { temperature: 1.0, top_k: 2, top_p: 1.0, ..Default::default() };
        for _ in 0..200 { let (t, _) = sample(&mut base.clone(), &p, &[], &mut rng); assert!(t == 0 || t == 1); }
        let p = SampleParams { temperature: 1.0, top_k: 0, top_p: 0.5, ..Default::default() };
        for _ in 0..200 { let (t, _) = sample(&mut base.clone(), &p, &[], &mut rng); assert_eq!(t, 0, "top-p 0.5 keeps only the 0.73-mass head"); }
    }

    #[test]
    fn repetition_penalty_demotes_recent_tokens() {
        let mut l = vec![2.0, 1.9];
        let p = SampleParams { temperature: 0.0, repeat_penalty: 1.5, ..Default::default() };
        assert_eq!(sample(&mut l, &p, &[0], &mut Rng::new(1)).0, 1);
    }

    #[test]
    fn sampling_is_reproducible_and_roughly_calibrated() {
        let base = vec![0.0f32, (2.0f32).ln()]; // probs 1/3, 2/3
        let p = SampleParams { temperature: 1.0, ..Default::default() };
        let run = |seed| { let mut rng = Rng::new(seed); (0..3000).filter(|_| sample(&mut base.clone(), &p, &[], &mut rng).0 == 1).count() };
        let a = run(7); assert_eq!(a, run(7));
        assert!((a as f32 / 3000.0 - 0.6667).abs() < 0.04, "got {a}/3000");
    }
}

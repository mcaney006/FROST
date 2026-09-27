// Per-call latency of the sampling kernels at vocabulary size (131072) — Mojo dylib vs Rust reference.
fn main() {
    let n = 131072usize;
    let mut s = 0x9e3779b97f4a7c15u64;
    let logits: Vec<f32> = (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0 }).collect();
    println!("backend = {} {:?}", frost_kernels::backend(), frost_kernels::load_error());
    for (name, f) in [
        ("topk k=64", Box::new(|l: &[f32]| { frost_kernels::topk(l, 64).unwrap().len() }) as Box<dyn Fn(&[f32]) -> usize>),
        ("softmax_topp n=131072 p=0.95", Box::new(|l: &[f32]| frost_kernels::softmax_topp(l, 0.7, 0.95).unwrap().len())),
        ("softmax_topp n=64 p=0.95", Box::new(|l: &[f32]| frost_kernels::softmax_topp(&l[..64], 0.7, 0.95).unwrap().len())),
        ("ref topk k=64", Box::new(|l: &[f32]| frost_kernels::reference::topk(l, 64).unwrap().len())),
        ("ref softmax_topp n=131072", Box::new(|l: &[f32]| frost_kernels::reference::softmax_topp(l, 0.7, 0.95).unwrap().len())),
    ] {
        let mut best = f64::MAX; let mut out = 0;
        for _ in 0..20 { let t = std::time::Instant::now(); out = f(&logits); best = best.min(t.elapsed().as_secs_f64() * 1e3); }
        println!("{name:34} best {best:8.3} ms  (returned {out})");
    }
}

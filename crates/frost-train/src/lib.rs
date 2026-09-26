//! frost-train: native training of small projection heads over the frozen
//! encoder embeddings. This is a REAL trained head (not a random init dressed
//! up as trained): a symmetric InfoNCE contrastive objective with
//! finite-difference-verified gradients, an optimizer-independent checkpoint,
//! and a train -> save -> reload -> rank cycle.
//!
//! Scope (honest): this trains ONE full-depth head (separate state/action
//! projectors into a shared space). It is NOT the progressive per-exit early
//! head — that needs intermediate encoder states and is not implemented here.

pub mod pipeline;

use serde::{Deserialize, Serialize};

/// Row-major matrix.
#[derive(Clone)]
pub struct Mat { pub r: usize, pub c: usize, pub v: Vec<f32> }
impl Mat {
    pub fn zeros(r: usize, c: usize) -> Mat { Mat { r, c, v: vec![0.0; r * c] } }
    pub fn at(&self, i: usize, j: usize) -> f32 { self.v[i * self.c + j] }
    pub fn set(&mut self, i: usize, j: usize, x: f32) { self.v[i * self.c + j] = x; }
    /// self [r,k] @ b [k,c] -> [r,c]
    pub fn matmul(&self, b: &Mat) -> Mat {
        assert_eq!(self.c, b.r);
        let mut o = Mat::zeros(self.r, b.c);
        for i in 0..self.r {
            for k in 0..self.c {
                let a = self.v[i * self.c + k];
                if a == 0.0 { continue; }
                for j in 0..b.c { o.v[i * b.c + j] += a * b.v[k * b.c + j]; }
            }
        }
        o
    }
    pub fn transpose(&self) -> Mat {
        let mut o = Mat::zeros(self.c, self.r);
        for i in 0..self.r { for j in 0..self.c { o.v[j * self.r + i] = self.v[i * self.c + j]; } }
        o
    }
}

/// Deterministic small PRNG (xorshift) for seeded init.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng { Rng(seed | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    pub fn normal(&mut self) -> f32 {
        // Box-Muller
        let u1 = (self.next() >> 11) as f32 / (1u64 << 53) as f32 + 1e-7;
        let u2 = (self.next() >> 11) as f32 / (1u64 << 53) as f32;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Two projection heads (state, action) into a shared P-dim comparison space.
#[derive(Serialize, Deserialize, Clone)]
pub struct Head {
    pub d: usize,
    pub p: usize,
    pub temp: f32,
    pub ws: Vec<f32>, // [d*p]
    pub wa: Vec<f32>, // [d*p]
}

impl Head {
    pub fn init(d: usize, p: usize, temp: f32, seed: u64) -> Head {
        let mut rng = Rng::new(seed);
        let scale = (1.0 / d as f32).sqrt();
        let gen = |rng: &mut Rng, n| (0..n).map(|_| rng.normal() * scale).collect::<Vec<_>>();
        Head { d, p, temp, ws: gen(&mut rng, d * p), wa: gen(&mut rng, d * p) }
    }
    fn ws_mat(&self) -> Mat { Mat { r: self.d, c: self.p, v: self.ws.clone() } }
    fn wa_mat(&self) -> Mat { Mat { r: self.d, c: self.p, v: self.wa.clone() } }

    /// Project + L2-normalize one embedding through the state head.
    pub fn project_state(&self, e: &[f32]) -> Vec<f32> { l2(proj(&self.ws, self.d, self.p, e)) }
    pub fn project_action(&self, e: &[f32]) -> Vec<f32> { l2(proj(&self.wa, self.d, self.p, e)) }

    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
    pub fn load(path: &std::path::Path) -> anyhow::Result<Head> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

fn proj(w: &[f32], d: usize, p: usize, e: &[f32]) -> Vec<f32> {
    let mut o = vec![0.0; p];
    for k in 0..d { let ek = e[k]; if ek == 0.0 { continue; } let row = k * p; for j in 0..p { o[j] += ek * w[row + j]; } }
    o
}
fn l2(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 { for x in &mut v { *x /= n; } }
    v
}

/// Symmetric InfoNCE loss + gradients over a batch. States `xs` [N,D], actions
/// `xa` [M,D] (first N are the positives for the N states; M-N are hard
/// negatives shared across the batch). Uses raw projected dot / temp as logits
/// (projections trained; cosine used at inference).
pub struct Batch { pub xs: Mat, pub xa: Mat }

pub fn loss_and_grad(head: &Head, b: &Batch) -> (f32, Vec<f32>, Vec<f32>) {
    let n = b.xs.r;
    let m = b.xa.r;
    let s = b.xs.matmul(&head.ws_mat()); // [N,P]
    let a = b.xa.matmul(&head.wa_mat()); // [M,P]
    // logits [N,M] = s @ a^T / temp
    let inv_t = 1.0 / head.temp;
    let mut logits = Mat::zeros(n, m);
    for i in 0..n { for j in 0..m {
        let mut d = 0.0; for k in 0..head.p { d += s.at(i, k) * a.at(j, k); }
        logits.set(i, j, d * inv_t);
    }}
    // row softmax CE (state->action), labels = i ; and col softmax CE (action->state) for first N.
    let mut dlog = Mat::zeros(n, m); // dL/dlogits
    let mut loss = 0.0;
    // rows
    for i in 0..n {
        let mut mx = f32::NEG_INFINITY; for j in 0..m { mx = mx.max(logits.at(i, j)); }
        let mut den = 0.0; for j in 0..m { den += (logits.at(i, j) - mx).exp(); }
        loss += -( (logits.at(i, i) - mx) - den.ln() );
        for j in 0..m {
            let p = (logits.at(i, j) - mx).exp() / den;
            dlog.set(i, j, dlog.at(i, j) + 0.5 * (p - if j == i { 1.0 } else { 0.0 }));
        }
    }
    // cols (only first N actions have a matching state)
    for j in 0..n {
        let mut mx = f32::NEG_INFINITY; for i in 0..n { mx = mx.max(logits.at(i, j)); }
        let mut den = 0.0; for i in 0..n { den += (logits.at(i, j) - mx).exp(); }
        loss += -( (logits.at(j, j) - mx) - den.ln() );
        for i in 0..n {
            let p = (logits.at(i, j) - mx).exp() / den;
            dlog.set(i, j, dlog.at(i, j) + 0.5 * (p - if i == j { 1.0 } else { 0.0 }));
        }
    }
    loss = 0.5 * loss / n as f32;
    // scale dlog by 1/N and by inv_t (chain through /temp)
    for x in &mut dlog.v { *x *= inv_t / n as f32; }
    // dS = dlog @ a ; dA = dlog^T @ s
    let ds = dlog.matmul(&a);              // [N,P]
    let da = dlog.transpose().matmul(&s);  // [M,P]
    // dWs = xs^T @ dS ; dWa = xa^T @ dA
    let dws = b.xs.transpose().matmul(&ds).v; // [D,P]
    let dwa = b.xa.transpose().matmul(&da).v;
    (loss, dws, dwa)
}

/// One full-batch gradient-descent step (SGD).
pub fn sgd_step(head: &mut Head, dws: &[f32], dwa: &[f32], lr: f32) {
    for (w, g) in head.ws.iter_mut().zip(dws) { *w -= lr * g; }
    for (w, g) in head.wa.iter_mut().zip(dwa) { *w -= lr * g; }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_mat(r: usize, c: usize, seed: u64) -> Mat {
        let mut rng = Rng::new(seed);
        Mat { r, c, v: (0..r * c).map(|_| rng.normal()).collect() }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        // tiny problem so central differences are cheap and exact-ish
        let (d, p, n, m) = (5usize, 4usize, 3usize, 5usize);
        let head = Head::init(d, p, 0.1, 42);
        let b = Batch { xs: rand_mat(n, d, 7), xa: rand_mat(m, d, 9) };
        let (_, dws, dwa) = loss_and_grad(&head, &b);
        let eps = 1e-3f32;
        let mut max_err = 0.0f32;
        // check a sample of Ws and Wa params
        for &idx in &[0usize, 3, 7, 11, 19] {
            for (which, analytic) in [(0u8, dws[idx]), (1u8, dwa[idx])] {
                let mut hp = head.clone();
                let w = if which == 0 { &mut hp.ws } else { &mut hp.wa };
                let orig = w[idx];
                w[idx] = orig + eps; let lp = loss_and_grad(&hp, &b).0;
                let w = if which == 0 { &mut hp.ws } else { &mut hp.wa };
                w[idx] = orig - eps; let lm = loss_and_grad(&hp, &b).0;
                let num = (lp - lm) / (2.0 * eps);
                let err = (num - analytic).abs() / (analytic.abs() + 1e-3);
                max_err = max_err.max(err);
            }
        }
        eprintln!("max relative grad error = {max_err:.5}");
        assert!(max_err < 1e-2, "analytic gradient must match finite differences (err {max_err})");
    }

    #[test]
    fn training_reduces_loss() {
        // separable synthetic: states and their positive actions share a signal
        let (d, p, n) = (16usize, 8usize, 6usize);
        let mut rng = Rng::new(123);
        let mut xs = Mat::zeros(n, d);
        let mut xa = Mat::zeros(n, d);
        for i in 0..n {
            for k in 0..d {
                let base = rng.normal();
                xs.set(i, k, base + 0.05 * rng.normal());
                xa.set(i, k, base + 0.05 * rng.normal()); // action correlates with its state
            }
        }
        let b = Batch { xs, xa };
        let mut head = Head::init(d, p, 1.0, 5);
        let l0 = loss_and_grad(&head, &b).0;
        for _ in 0..300 {
            let (_, dws, dwa) = loss_and_grad(&head, &b);
            sgd_step(&mut head, &dws, &dwa, 0.05);
        }
        let l1 = loss_and_grad(&head, &b).0;
        eprintln!("loss {l0:.4} -> {l1:.4}");
        assert!(l1 < l0 * 0.5, "training must substantially reduce contrastive loss");
    }
}

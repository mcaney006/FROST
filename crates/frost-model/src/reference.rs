//! Independent scalar (f64, CPU, pure-Rust) reference forward for the nomic-bert
//! encoder. Used to parity-check the MLX path and to isolate architecture vs
//! composition bugs. Slow and simple on purpose — no MLX, no FFI.

use frost_tensors::SafeTensors;
use std::path::Path;

pub struct Ref {
    pub h: usize,
    pub layers: usize,
    pub heads: usize,
    pub hd: usize,
    pub inter: usize,
    pub eps: f64,
    pub rope_base: f64,
    word_emb: Vec<f32>,
    tt0: Vec<f64>,
    emb_ln_w: Vec<f64>,
    emb_ln_b: Vec<f64>,
    l: Vec<RefLayer>,
}

struct RefLayer {
    wqkv: Vec<f64>, // [3h, h] row-major (as stored)
    wo: Vec<f64>,   // [h,h]
    fc11: Vec<f64>, // [inter,h]
    fc12: Vec<f64>,
    fc2: Vec<f64>,  // [h,inter]
    n1w: Vec<f64>, n1b: Vec<f64>, n2w: Vec<f64>, n2b: Vec<f64>,
}

fn f64v(st: &SafeTensors, name: &str) -> Vec<f64> {
    st.f32(name).unwrap().0.iter().map(|&x| x as f64).collect()
}

impl Ref {
    pub fn load(dir: &Path) -> anyhow::Result<Ref> {
        let cfg = crate::model::Config::from_json(&dir.join("config.json"))?;
        let st = SafeTensors::open(&dir.join("model.safetensors"))?;
        let (word_emb, _) = st.f32("embeddings.word_embeddings.weight")?;
        let tt: Vec<f64> = f64v(&st, "embeddings.token_type_embeddings.weight");
        let mut l = Vec::new();
        for i in 0..cfg.layers {
            let p = format!("encoder.layers.{i}");
            l.push(RefLayer {
                wqkv: f64v(&st, &format!("{p}.attn.Wqkv.weight")),
                wo: f64v(&st, &format!("{p}.attn.out_proj.weight")),
                fc11: f64v(&st, &format!("{p}.mlp.fc11.weight")),
                fc12: f64v(&st, &format!("{p}.mlp.fc12.weight")),
                fc2: f64v(&st, &format!("{p}.mlp.fc2.weight")),
                n1w: f64v(&st, &format!("{p}.norm1.weight")),
                n1b: f64v(&st, &format!("{p}.norm1.bias")),
                n2w: f64v(&st, &format!("{p}.norm2.weight")),
                n2b: f64v(&st, &format!("{p}.norm2.bias")),
            });
        }
        Ok(Ref {
            h: cfg.hidden, layers: cfg.layers, heads: cfg.heads, hd: cfg.head_dim,
            inter: cfg.intermediate, eps: cfg.eps as f64, rope_base: cfg.rope_base as f64,
            word_emb, tt0: tt[0..cfg.hidden].to_vec(),
            emb_ln_w: f64v(&st, "emb_ln.weight"), emb_ln_b: f64v(&st, "emb_ln.bias"), l,
        })
    }

    fn layer_norm(&self, row: &mut [f64], w: &[f64], b: &[f64]) {
        let n = row.len();
        let mean = row.iter().sum::<f64>() / n as f64;
        let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + self.eps).sqrt();
        for j in 0..n { row[j] = (row[j] - mean) * inv * w[j] + b[j]; }
    }

    // y[out] = x[in] @ W[out,in]^T  (PyTorch Linear)
    fn linear(&self, x: &[f64], w: &[f64], inp: usize, out: usize) -> Vec<f64> {
        let mut y = vec![0.0; out];
        for o in 0..out {
            let wr = &w[o * inp..(o + 1) * inp];
            let mut acc = 0.0;
            for i in 0..inp { acc += x[i] * wr[i]; }
            y[o] = acc;
        }
        y
    }

    /// Non-interleaved (NeoX) RoPE on a [hd] head vector at position `pos`.
    fn rope(&self, v: &mut [f64], pos: usize) {
        let hd = self.hd;
        let half = hd / 2;
        for i in 0..half {
            let theta = pos as f64 / self.rope_base.powf(2.0 * i as f64 / hd as f64);
            let (s, c) = theta.sin_cos();
            let a = v[i];
            let b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }

    /// Full forward -> L2-normalized pooled embedding.
    pub fn embed(&self, ids: &[u32]) -> Vec<f64> {
        let (s, h, hd, nh) = (ids.len(), self.h, self.hd, self.heads);
        // embeddings + emb_ln
        let mut x = vec![vec![0.0f64; h]; s];
        for t in 0..s {
            let row = ids[t] as usize * h;
            for j in 0..h { x[t][j] = self.word_emb[row + j] as f64 + self.tt0[j]; }
            self.layer_norm(&mut x[t], &self.emb_ln_w, &self.emb_ln_b);
        }
        for lyr in &self.l {
            // ---- attention ----
            // project qkv per token
            let mut q = vec![vec![0.0; h]; s];
            let mut k = vec![vec![0.0; h]; s];
            let mut v = vec![vec![0.0; h]; s];
            for t in 0..s {
                let qkv = self.linear(&x[t], &lyr.wqkv, h, 3 * h);
                q[t].copy_from_slice(&qkv[0..h]);
                k[t].copy_from_slice(&qkv[h..2 * h]);
                v[t].copy_from_slice(&qkv[2 * h..3 * h]);
            }
            // rope per head
            for t in 0..s {
                for head in 0..nh {
                    self.rope(&mut q[t][head * hd..(head + 1) * hd], t);
                    self.rope(&mut k[t][head * hd..(head + 1) * hd], t);
                }
            }
            // attention per head, bidirectional, scale 1/sqrt(hd)
            let scale = 1.0 / (hd as f64).sqrt();
            let mut ctx = vec![vec![0.0; h]; s];
            for head in 0..nh {
                let off = head * hd;
                for ti in 0..s {
                    let mut scores = vec![0.0; s];
                    let mut mx = f64::NEG_INFINITY;
                    for tj in 0..s {
                        let mut dot = 0.0;
                        for d in 0..hd { dot += q[ti][off + d] * k[tj][off + d]; }
                        scores[tj] = dot * scale;
                        if scores[tj] > mx { mx = scores[tj]; }
                    }
                    let mut den = 0.0;
                    for tj in 0..s { scores[tj] = (scores[tj] - mx).exp(); den += scores[tj]; }
                    for d in 0..hd {
                        let mut acc = 0.0;
                        for tj in 0..s { acc += scores[tj] / den * v[tj][off + d]; }
                        ctx[ti][off + d] = acc;
                    }
                }
            }
            // out proj + residual + norm1
            for t in 0..s {
                let o = self.linear(&ctx[t], &lyr.wo, h, h);
                for j in 0..h { x[t][j] += o[j]; }
                self.layer_norm(&mut x[t], &lyr.n1w, &lyr.n1b);
            }
            // SwiGLU MLP + residual + norm2
            for t in 0..s {
                let g = self.linear(&x[t], &lyr.fc11, h, self.inter);
                let up = self.linear(&x[t], &lyr.fc12, h, self.inter);
                let mut hid = vec![0.0; self.inter];
                for j in 0..self.inter {
                    let gate = up[j] / (1.0 + (-up[j]).exp()); // silu on fc12 (gate)
                    hid[j] = g[j] * gate;                       // value fc11 * gated
                }
                let m = self.linear(&hid, &lyr.fc2, self.inter, h);
                for j in 0..h { x[t][j] += m[j]; }
                self.layer_norm(&mut x[t], &lyr.n2w, &lyr.n2b);
            }
        }
        // mean pool + L2
        let mut pooled = vec![0.0; h];
        for t in 0..s { for j in 0..h { pooled[j] += x[t][j]; } }
        for j in 0..h { pooled[j] /= s as f64; }
        let nrm = pooled.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 0.0 { for j in 0..h { pooled[j] /= nrm; } }
        pooled
    }
}

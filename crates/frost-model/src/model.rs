//! nomic-bert encoder: real weights (safetensors) + real forward pass on MLX.
//!
//! Architecture (read from config.json, never hardcoded blindly):
//!   token_type + word embeddings -> emb LayerNorm -> N postnorm blocks
//!   block: attn(RoPE, bidirectional) -> LN(x+attn) -> SwiGLU MLP -> LN(x+mlp)
//!   pool: masked mean over tokens ; caller L2-normalizes.
//! rope base=1000, non-interleaved; no learned positions; no qkv/fc1 bias.

use crate::mlx::{Arr, Stream};
use anyhow::{Context, Result};
use frost_tensors::SafeTensors;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_traditional: bool,
}

impl Config {
    pub fn from_json(path: &Path) -> Result<Config> {
        let v: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
        let g = |k: &str| v.get(k).and_then(|x| x.as_u64());
        let hidden = g("hidden_size").context("hidden_size")? as usize;
        let heads = g("num_attention_heads").context("heads")? as usize;
        Ok(Config {
            hidden,
            layers: g("num_hidden_layers").context("layers")? as usize,
            heads,
            head_dim: hidden / heads,
            intermediate: g("intermediate_size").context("intermediate")? as usize,
            vocab: g("vocab_size").context("vocab")? as usize,
            eps: v.get("layer_norm_epsilon").and_then(|x| x.as_f64()).unwrap_or(1e-12) as f32,
            rope_base: v.get("rotary_emb_base").and_then(|x| x.as_f64()).unwrap_or(1000.0) as f32,
            rope_traditional: v.get("rotary_emb_interleaved").and_then(|x| x.as_bool()).unwrap_or(false),
        })
    }
}

struct Layer {
    wq: Arr, wk: Arr, wv: Arr, // transposed [in,out]
    wo: Arr,
    fc11: Arr, fc12: Arr, fc2: Arr, // transposed
    n1w: Arr, n1b: Arr, n2w: Arr, n2b: Arr,
}

pub struct Model {
    pub cfg: Config,
    s: Stream,
    word_emb: Vec<f32>,      // [vocab, hidden] for Rust-side gather
    token_type0: Vec<f32>,   // [hidden] token_type embedding row 0
    emb_ln_w: Arr,
    emb_ln_b: Arr,
    layers: Vec<Layer>,
    pub fingerprint: String,
}

impl Model {
    pub fn load(dir: &Path, revision: &str) -> Result<Model> {
        let cfg = Config::from_json(&dir.join("config.json"))?;
        let st = SafeTensors::open(&dir.join("model.safetensors"))?;
        let s = Stream::gpu();
        let h = cfg.hidden as i32;

        // transposed projection: raw is [out,in] row-major -> Arr [out,in] -> transpose -> [in,out]
        let load_t = |name: &str, out: i32, inp: i32| -> Result<Arr> {
            let (v, shp) = st.f32(name)?;
            assert_eq!(shp, vec![out as usize, inp as usize], "shape of {name}");
            let a = Arr::from_f32(&v, &[out, inp]);
            let t = a.transpose(&[1, 0], &s);
            t.eval();
            Ok(t)
        };
        let load_vec = |name: &str, n: i32| -> Result<Arr> {
            let (v, _) = st.f32(name)?;
            let a = Arr::from_f32(&v, &[n]);
            a.eval();
            Ok(a)
        };

        let (word_emb, wshape) = st.f32("embeddings.word_embeddings.weight")?;
        assert_eq!(wshape, vec![cfg.vocab, cfg.hidden]);
        let (tt, _) = st.f32("embeddings.token_type_embeddings.weight")?;
        let token_type0 = tt[0..cfg.hidden].to_vec();

        let emb_ln_w = load_vec("emb_ln.weight", h)?;
        let emb_ln_b = load_vec("emb_ln.bias", h)?;

        let inter = cfg.intermediate as i32;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("encoder.layers.{i}");
            // fused Wqkv [3h, h] -> split rows into q,k,v [h,h] then transpose each
            let (qkv, qshape) = st.f32(&format!("{p}.attn.Wqkv.weight"))?;
            assert_eq!(qshape, vec![3 * cfg.hidden, cfg.hidden]);
            let hh = cfg.hidden;
            let mk_t = |rows: &[f32]| {
                let a = Arr::from_f32(rows, &[h, h]);
                let t = a.transpose(&[1, 0], &s);
                t.eval();
                t
            };
            let wq = mk_t(&qkv[0..hh * hh]);
            let wk = mk_t(&qkv[hh * hh..2 * hh * hh]);
            let wv = mk_t(&qkv[2 * hh * hh..3 * hh * hh]);
            layers.push(Layer {
                wq, wk, wv,
                wo: load_t(&format!("{p}.attn.out_proj.weight"), h, h)?,
                fc11: load_t(&format!("{p}.mlp.fc11.weight"), inter, h)?,
                fc12: load_t(&format!("{p}.mlp.fc12.weight"), inter, h)?,
                fc2: load_t(&format!("{p}.mlp.fc2.weight"), h, inter)?,
                n1w: load_vec(&format!("{p}.norm1.weight"), h)?,
                n1b: load_vec(&format!("{p}.norm1.bias"), h)?,
                n2w: load_vec(&format!("{p}.norm2.weight"), h)?,
                n2b: load_vec(&format!("{p}.norm2.bias"), h)?,
            });
        }

        let fingerprint = format!(
            "nomic-embed-text-v1.5@{revision}:h{}:l{}:heads{}:rope{}",
            cfg.hidden, cfg.layers, cfg.heads, cfg.rope_base
        );
        Ok(Model { cfg, s, word_emb, token_type0, emb_ln_w, emb_ln_b, layers, fingerprint })
    }

    /// Embeddings (Rust-side gather + token_type[0]) -> emb LayerNorm -> [1,S,H].
    fn embed(&self, ids: &[u32]) -> Arr {
        let s = &self.s;
        let (seq, h) = (ids.len(), self.cfg.hidden);
        let mut emb = vec![0f32; seq * h];
        for (t, &id) in ids.iter().enumerate() {
            let row = (id as usize) * h;
            let dst = &mut emb[t * h..(t + 1) * h];
            dst.copy_from_slice(&self.word_emb[row..row + h]);
            for j in 0..h { dst[j] += self.token_type0[j]; }
        }
        Arr::from_f32(&emb, &[seq as i32, h as i32])
            .layer_norm(&self.emb_ln_w, &self.emb_ln_b, self.cfg.eps, s)
            .reshape(&[1, seq as i32, h as i32], s)
    }

    /// One postnorm encoder block. The single definition every forward path uses:
    /// attn(RoPE, bidirectional) -> LN(x+attn) -> SwiGLU `fc11 * silu(fc12)` -> LN(x+mlp).
    fn block(&self, x: &Arr, l: &Layer, sq: i32) -> Arr {
        let s = &self.s;
        let h = self.cfg.hidden as i32;
        let (hd, nh) = (self.cfg.head_dim as i32, self.cfg.heads as i32);
        let scale = Arr::scalar(1.0 / (self.cfg.head_dim as f32).sqrt());
        let q = x.matmul(&l.wq, s).reshape(&[1, sq, nh, hd], s).transpose(&[0, 2, 1, 3], s)
            .rope(hd, self.cfg.rope_traditional, self.cfg.rope_base, s);
        let k = x.matmul(&l.wk, s).reshape(&[1, sq, nh, hd], s).transpose(&[0, 2, 1, 3], s)
            .rope(hd, self.cfg.rope_traditional, self.cfg.rope_base, s);
        let v = x.matmul(&l.wv, s).reshape(&[1, sq, nh, hd], s).transpose(&[0, 2, 1, 3], s);
        let probs = q.matmul(&k.transpose(&[0, 1, 3, 2], s), s).mul(&scale, s).softmax_last(s);
        let ctx = probs.matmul(&v, s).transpose(&[0, 2, 1, 3], s).reshape(&[1, sq, h], s);
        let x = x.add(&ctx.matmul(&l.wo, s), s).reshape(&[sq, h], s)
            .layer_norm(&l.n1w, &l.n1b, self.cfg.eps, s).reshape(&[1, sq, h], s);
        let hmid = x.matmul(&l.fc11, s).mul(&x.matmul(&l.fc12, s).silu(s), s);
        x.add(&hmid.matmul(&l.fc2, s), s).reshape(&[sq, h], s)
            .layer_norm(&l.n2w, &l.n2b, self.cfg.eps, s).reshape(&[1, sq, h], s)
    }

    /// Forward through the first `nlayers` blocks -> pooled (un-normalized) [hidden].
    pub fn forward_upto(&self, ids: &[u32], nlayers: usize) -> Vec<f32> {
        let sq = ids.len() as i32;
        let mut x = self.embed(ids);
        for l in self.layers.iter().take(nlayers) { x = self.block(&x, l, sq); }
        x.mean_axis(1, false, &self.s).to_vec() // all tokens valid; batch=1
    }

    /// Full-depth forward for one token-id sequence -> pooled (un-normalized) [hidden].
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> { self.forward_upto(ids, self.layers.len()) }
}

/// L2-normalize in place, returning the vector.
pub fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let ss: f32 = v.iter().map(|x| x * x).sum();
    if ss > 0.0 {
        let inv = 1.0 / ss.sqrt();
        for x in &mut v { *x *= inv; }
    }
    v
}

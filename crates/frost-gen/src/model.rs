//! Ministral-3 text decoder on MLX (text-only path; the vision tower is never loaded).
//!
//! Architecture (from the pinned `config.json` `text_config`, validated, not assumed):
//!   embed (4-bit affine) -> N x [ RMSNorm -> GQA attention (YaRN RoPE, llama-4 attention
//!   scale) -> residual -> RMSNorm -> SwiGLU MLP -> residual ] -> RMSNorm -> lm_head (4-bit).
//! Every projection is an MLX affine-quantized matmul over the checkpoint's packed U32
//! weights with per-group bf16 scales/biases. Activations are bf16; logits are read as f32.

use frost_mlx::{Arr, Dtype, Stream};
use frost_tensors::{SafeTensors, SafetensorsError, Dtype as StDtype};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("config.json: {0}")]
    Config(String),
    #[error("weights: {0}")]
    Weights(#[from] SafetensorsError),
    #[error("weights: tensor {0} not present in any shard")]
    MissingTensor(String),
    #[error("unsupported checkpoint: {0}")]
    Unsupported(String),
    #[error("not enough free memory to load the model: {needed_mib} MiB needed (weights + working set), {available_mib} MiB available; close other apps or another FROST instance")]
    Memory { needed_mib: u64, available_mib: u64 },
}

/// Working-set headroom reserved on top of the weights: KV cache at the 4096-token budget
/// (~570 MiB for 34 layers x 8 kv heads x 128 x bf16) plus transient activations.
pub const LOAD_HEADROOM_BYTES: u64 = 768 * 1024 * 1024;

/// Text-model hyperparameters read from `config.json`.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub yarn_factor: f64,
    pub yarn_original_max: usize,
    pub yarn_beta_fast: f64,
    pub yarn_beta_slow: f64,
    pub yarn_mscale: f64,
    pub yarn_mscale_all_dim: f64,
    pub llama4_beta: f64,
    pub group_size: usize,
    pub bits: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn from_json(path: &Path) -> Result<Config, ModelError> {
        let text = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| ModelError::Config(e.to_string()))?;
        let arch = v["architectures"][0].as_str().unwrap_or("");
        if arch != "Mistral3ForConditionalGeneration" {
            return Err(ModelError::Unsupported(format!("architecture {arch:?}; this loader implements Mistral3ForConditionalGeneration (text path)")));
        }
        let t = &v["text_config"];
        if t["model_type"].as_str() != Some("ministral3") {
            return Err(ModelError::Unsupported(format!("text_config.model_type {:?}; expected ministral3", t["model_type"])));
        }
        let q = if v["quantization"].is_object() { &v["quantization"] } else { &v["quantization_config"] };
        if q["mode"].as_str().unwrap_or("affine") != "affine" {
            return Err(ModelError::Unsupported(format!("quantization mode {:?}", q["mode"])));
        }
        let u = |o: &serde_json::Value, k: &str| o[k].as_u64().map(|x| x as usize).ok_or_else(|| ModelError::Config(format!("missing {k}")));
        let f = |o: &serde_json::Value, k: &str| o[k].as_f64().ok_or_else(|| ModelError::Config(format!("missing {k}")));
        let rp = &t["rope_parameters"];
        let rope_type = rp["rope_type"].as_str().or(rp["type"].as_str()).unwrap_or("default");
        if rope_type != "yarn" {
            return Err(ModelError::Unsupported(format!("rope type {rope_type:?}; expected yarn")));
        }
        let heads = u(t, "num_attention_heads")?;
        let hidden = u(t, "hidden_size")?;
        let cfg = Config {
            hidden,
            layers: u(t, "num_hidden_layers")?,
            heads,
            kv_heads: t["num_key_value_heads"].as_u64().map(|x| x as usize).unwrap_or(heads),
            head_dim: t["head_dim"].as_u64().map(|x| x as usize).unwrap_or(hidden / heads),
            intermediate: u(t, "intermediate_size")?,
            vocab: u(t, "vocab_size")?,
            rms_eps: f(t, "rms_norm_eps")? as f32,
            rope_theta: f(rp, "rope_theta")?,
            yarn_factor: f(rp, "factor")?,
            yarn_original_max: u(rp, "original_max_position_embeddings")?,
            yarn_beta_fast: rp["beta_fast"].as_f64().unwrap_or(32.0),
            yarn_beta_slow: rp["beta_slow"].as_f64().unwrap_or(1.0),
            yarn_mscale: rp["mscale"].as_f64().unwrap_or(1.0),
            yarn_mscale_all_dim: rp["mscale_all_dim"].as_f64().unwrap_or(0.0),
            llama4_beta: rp["llama_4_scaling_beta"].as_f64().unwrap_or(0.0),
            group_size: u(q, "group_size")?,
            bits: u(q, "bits")?,
            tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        };
        if cfg.bits != 4 || cfg.group_size == 0 || cfg.hidden % cfg.group_size != 0 || cfg.intermediate % cfg.group_size != 0 {
            return Err(ModelError::Unsupported(format!("quantization bits={} group_size={}", cfg.bits, cfg.group_size)));
        }
        if cfg.heads % cfg.kv_heads != 0 || cfg.head_dim % 2 != 0 {
            return Err(ModelError::Unsupported(format!("heads={} kv_heads={} head_dim={}", cfg.heads, cfg.kv_heads, cfg.head_dim)));
        }
        Ok(cfg)
    }

    /// YaRN per-pair rotary denominators (`freqs` for `mlx_fast_rope`), mirroring the upstream
    /// mlx-lm `YarnRoPE` construction; computed in f64 and stored as f32.
    pub fn yarn_freqs(&self) -> Vec<f32> {
        let dims = self.head_dim as f64;
        let base = self.rope_theta;
        let half = self.head_dim / 2;
        let find_correction_dim = |num_rotations: f64| {
            dims * (self.yarn_original_max as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
        };
        let low = find_correction_dim(self.yarn_beta_fast).floor().max(0.0);
        let high = find_correction_dim(self.yarn_beta_slow).ceil().min(dims - 1.0);
        let (lo, mut hi) = (low, high);
        if (lo - hi).abs() < f64::EPSILON { hi += 0.001; }
        (0..half).map(|i| {
            let freq_extra = base.powf(2.0 * i as f64 / dims);
            let freq_inter = self.yarn_factor * freq_extra;
            let ramp = ((i as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            ((freq_inter * freq_extra) / (freq_inter * mask + freq_extra * (1.0 - mask))) as f32
        }).collect()
    }

    /// YaRN attention-magnitude scale applied to q/k before RoPE (1.0 when mscale == mscale_all_dim).
    pub fn yarn_mscale(&self) -> f32 {
        let get = |m: f64| if self.yarn_factor <= 1.0 { 1.0 } else { 0.1 * m * self.yarn_factor.ln() + 1.0 };
        (get(self.yarn_mscale) / get(self.yarn_mscale_all_dim)) as f32
    }

    /// llama-4 style query scale for absolute positions `offset .. offset+len`.
    pub fn attn_scales(&self, offset: usize, len: usize) -> Vec<f32> {
        (0..len).map(|i| {
            let pos = (offset + i) as f64;
            (1.0 + self.llama4_beta * (1.0 + (pos / self.yarn_original_max as f64).floor()).ln()) as f32
        }).collect()
    }
}

/// One affine-quantized projection: `w` U32 [out, in/8], `scales`/`biases` bf16 [out, in/group].
pub struct QLinear { pub w: Arr, pub scales: Arr, pub biases: Arr, pub out: usize, pub inp: usize }

struct Layer {
    ln1: Arr, ln2: Arr,
    q: QLinear, k: QLinear, v: QLinear, o: QLinear,
    gate: QLinear, up: QLinear, down: QLinear,
}

/// Per-layer key/value cache: preallocated in `STEP`-token slabs, trimmable for prefix reuse.
pub struct KvCache { keys: Vec<Option<Arr>>, values: Vec<Option<Arr>>, pub offset: usize }
const STEP: i32 = 256;

impl KvCache {
    fn new(layers: usize) -> KvCache {
        KvCache { keys: (0..layers).map(|_| None).collect(), values: (0..layers).map(|_| None).collect(), offset: 0 }
    }
    /// Drop everything after `n` tokens (the arrays keep their capacity).
    pub fn trim_to(&mut self, n: usize) { self.offset = self.offset.min(n); }
    pub fn clear(&mut self) { for k in &mut self.keys { *k = None; } for v in &mut self.values { *v = None; } self.offset = 0; }
    /// Bytes held by the cache arrays (capacity, not just the used prefix).
    pub fn nbytes(&self) -> usize {
        self.keys.iter().chain(self.values.iter()).filter_map(|a| a.as_ref()).map(|a| a.nbytes()).sum()
    }

    fn update(&mut self, layer: usize, k: &Arr, v: &Arr, prev: usize, s: &Stream) -> (Arr, Arr) {
        let shape = k.shape(); // [1, kvh, L, hd]
        let (b, h, l) = (shape[0], shape[1], shape[2]);
        let need = prev as i32 + l;
        let cap = self.keys[layer].as_ref().map(|a| a.shape()[2]).unwrap_or(0);
        if need > cap {
            // Extend by whole STEP slabs past the live prefix (stale tail beyond `prev` is dropped).
            let steps = (l + STEP - 1) / STEP;
            let grow = |cur: Option<&Arr>, d: i32, dtype: Dtype| -> Arr {
                let fresh = Arr::zeros(&[b, h, steps * STEP, d], dtype, s);
                match cur {
                    None => fresh,
                    Some(existing) => Arr::concat(&[&existing.slice(&[0, 0, 0, 0], &[b, h, prev as i32, d], s), &fresh], 2, s),
                }
            };
            self.keys[layer] = Some(grow(self.keys[layer].as_ref(), shape[3], k.dtype()));
            self.values[layer] = Some(grow(self.values[layer].as_ref(), v.shape()[3], v.dtype()));
        }
        let write = |dst: &Arr, upd: &Arr| dst.slice_update(upd, &[0, 0, prev as i32, 0], &[b, h, need, upd.shape()[3]], s);
        let nk = write(self.keys[layer].as_ref().expect("allocated"), k);
        let nv = write(self.values[layer].as_ref().expect("allocated"), v);
        let kk = nk.slice(&[0, 0, 0, 0], &[b, h, need, nk.shape()[3]], s);
        let vv = nv.slice(&[0, 0, 0, 0], &[b, h, need, nv.shape()[3]], s);
        self.keys[layer] = Some(nk);
        self.values[layer] = Some(nv);
        (kk, vv)
    }
}

pub struct Model {
    pub cfg: Config,
    pub s: Stream,
    embed: QLinear,
    layers: Vec<Layer>,
    norm: Arr,
    lm_head: QLinear,
    freqs: Arr,
    yarn_mscale: f32,
    /// Bytes copied into MLX for weights (reported separately from the KV cache).
    pub weight_bytes: usize,
    /// What the Zig layout validator saw at load (tensor count, non-finite scale/bias count).
    pub layout_check: String,
}

/// Reinterpret bytes as `T` without copying when the slice is aligned and sized for it.
fn view<T: Copy>(bytes: &[u8]) -> Option<&[T]> {
    let n = std::mem::size_of::<T>();
    if bytes.len() % n != 0 || (bytes.as_ptr() as usize) % std::mem::align_of::<T>() != 0 { return None; }
    Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / n) })
}

/// Validate one affine-quantized projection's packed layout and scale/bias finiteness with the
/// Zig kernel (`frost_index::q4_validate`). Aligned shard bytes are viewed in place; an unaligned
/// shard falls back to a copy so the check still runs.
fn validate_q4(shards: &Shards, name: &str, out: usize, packed: usize, groups: usize, group_size: usize, bits: usize) -> Result<frost_index::Q4Stats, ModelError> {
    let wb = shards.bytes(&format!("{name}.weight"), StDtype::U32, &[out, packed])?;
    let sb = shards.bytes(&format!("{name}.scales"), StDtype::BF16, &[out, groups])?;
    let bb = shards.bytes(&format!("{name}.biases"), StDtype::BF16, &[out, groups])?;
    let run = |w: &[u32], s: &[u16], b: &[u16]| frost_index::q4_validate(w, out, packed, s, b, groups, group_size as u32, bits as u32)
        .map_err(|e| ModelError::Unsupported(format!("{name}: quantized layout rejected by the Zig validator: {e}")));
    match (view::<u32>(wb), view::<u16>(sb), view::<u16>(bb)) {
        (Some(w), Some(s), Some(b)) => run(w, s, b),
        _ => {
            let w: Vec<u32> = wb.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let s: Vec<u16> = sb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let b: Vec<u16> = bb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            run(&w, &s, &b)
        }
    }
}

/// A set of safetensors shards searched in order for each tensor name.
pub struct Shards { pub files: Vec<(PathBuf, SafeTensors)> }
impl Shards {
    pub fn open(dir: &Path) -> Result<Shards, ModelError> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        paths.sort();
        if paths.is_empty() { return Err(ModelError::Unsupported(format!("no .safetensors shards in {}", dir.display()))); }
        let mut files = Vec::new();
        for p in paths { let st = SafeTensors::open(&p)?; files.push((p, st)); }
        Ok(Shards { files })
    }
    fn find(&self, name: &str) -> Result<&SafeTensors, ModelError> {
        self.files.iter().map(|(_, st)| st).find(|st| st.contains(name)).ok_or_else(|| ModelError::MissingTensor(name.to_string()))
    }
    /// Raw bytes of a tensor of exactly `dtype`/`shape`.
    pub fn bytes(&self, name: &str, dtype: StDtype, shape: &[usize]) -> Result<&[u8], ModelError> {
        Ok(self.find(name)?.expect(name, dtype, shape)?)
    }
    /// Copy a tensor of exactly `dtype`/`shape` into an MLX array.
    pub fn arr(&self, name: &str, dtype: StDtype, shape: &[usize], mlx_dtype: Dtype) -> Result<Arr, ModelError> {
        let st = self.find(name)?;
        let bytes = st.expect(name, dtype, shape)?;
        let ishape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        Ok(Arr::from_bytes(bytes, &ishape, mlx_dtype))
    }
    pub fn names(&self) -> impl Iterator<Item = &str> { self.files.iter().flat_map(|(_, st)| st.names()) }
}

impl Model {
    pub fn load(dir: &Path) -> Result<Model, ModelError> {
        let cfg = Config::from_json(&dir.join("config.json"))?;
        let shards = Shards::open(dir)?;
        // Admission: refuse to load into a machine that would have to swap (the text-model tensors
        // are copied into MLX buffers, so the planned bytes are known from the headers alone).
        let planned: u64 = shards.files.iter().flat_map(|(_, st)| st.names().filter(|n| n.starts_with("language_model.")).filter_map(|n| st.info(n)).map(|i| i.nbytes() as u64)).sum();
        let needed = planned + LOAD_HEADROOM_BYTES;
        if let Some(avail) = frost_platform::available_memory_bytes() {
            // Buffers MLX already holds in its cache are reusable by this load, so they count.
            let avail = avail + frost_mlx::memory().cache as u64;
            if avail < needed { return Err(ModelError::Memory { needed_mib: needed >> 20, available_mib: avail >> 20 }); }
        }
        let s = Stream::gpu();
        let mut weight_bytes = 0usize;

        let mut q4_tensors = 0usize;
        let mut q4_nonfinite = 0u64;
        let mut qlin = |name: &str, out: usize, inp: usize| -> Result<QLinear, ModelError> {
            let packed = inp / (32 / cfg.bits);
            let groups = inp / cfg.group_size;
            // Zig-owned layout validation over the raw checkpoint bytes before anything is copied to MLX.
            let st = validate_q4(&shards, name, out, packed, groups, cfg.group_size, cfg.bits)?;
            q4_tensors += 1; q4_nonfinite += st.nonfinite;
            let w = shards.arr(&format!("{name}.weight"), StDtype::U32, &[out, packed], Dtype::U32)?;
            let scales = shards.arr(&format!("{name}.scales"), StDtype::BF16, &[out, groups], Dtype::BF16)?;
            let biases = shards.arr(&format!("{name}.biases"), StDtype::BF16, &[out, groups], Dtype::BF16)?;
            weight_bytes += w.nbytes() + scales.nbytes() + biases.nbytes();
            Ok(QLinear { w, scales, biases, out, inp })
        };
        let h = cfg.hidden;
        let embed = qlin("language_model.model.embed_tokens", cfg.vocab, h)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("language_model.model.layers.{i}");
            layers.push(Layer {
                q: qlin(&format!("{p}.self_attn.q_proj"), cfg.heads * cfg.head_dim, h)?,
                k: qlin(&format!("{p}.self_attn.k_proj"), cfg.kv_heads * cfg.head_dim, h)?,
                v: qlin(&format!("{p}.self_attn.v_proj"), cfg.kv_heads * cfg.head_dim, h)?,
                o: qlin(&format!("{p}.self_attn.o_proj"), h, cfg.heads * cfg.head_dim)?,
                gate: qlin(&format!("{p}.mlp.gate_proj"), cfg.intermediate, h)?,
                up: qlin(&format!("{p}.mlp.up_proj"), cfg.intermediate, h)?,
                down: qlin(&format!("{p}.mlp.down_proj"), h, cfg.intermediate)?,
                ln1: shards.arr(&format!("{p}.input_layernorm.weight"), StDtype::BF16, &[h], Dtype::BF16)?,
                ln2: shards.arr(&format!("{p}.post_attention_layernorm.weight"), StDtype::BF16, &[h], Dtype::BF16)?,
            });
        }
        let norm = shards.arr("language_model.model.norm.weight", StDtype::BF16, &[h], Dtype::BF16)?;
        if cfg.tie_word_embeddings {
            return Err(ModelError::Unsupported("tied embeddings are not implemented for this checkpoint family".into()));
        }
        let lm_head = qlin("language_model.lm_head", cfg.vocab, h)?;
        drop(qlin);
        weight_bytes += (cfg.layers * 2 + 1) * h * 2; // RMSNorm weights (bf16)
        let layout_check = format!("zig q4_validate: {q4_tensors} tensors, {q4_nonfinite} non-finite scales/biases");
        let freqs = Arr::from_f32(&cfg.yarn_freqs(), &[(cfg.head_dim / 2) as i32]);
        let yarn_mscale = cfg.yarn_mscale();
        // Materialize all weights now so load time and memory are accounted before the first request.
        let mut all: Vec<&Arr> = vec![&embed.w, &embed.scales, &embed.biases, &norm, &lm_head.w, &lm_head.scales, &lm_head.biases, &freqs];
        for l in &layers {
            for q in [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down] { all.push(&q.w); all.push(&q.scales); all.push(&q.biases); }
            all.push(&l.ln1); all.push(&l.ln2);
        }
        frost_mlx::eval_all(&all);
        Ok(Model { cfg, s, embed, layers, norm, lm_head, freqs, yarn_mscale, weight_bytes, layout_check })
    }

    pub fn new_cache(&self) -> KvCache { KvCache::new(self.cfg.layers) }

    fn proj(&self, x: &Arr, l: &QLinear) -> Arr {
        x.quantized_matmul(&l.w, &l.scales, &l.biases, self.cfg.group_size as i32, self.cfg.bits as i32, &self.s)
    }

    /// Embedding lookup: gather the quantized rows then dequantize (QuantizedEmbedding semantics).
    fn embed(&self, ids: &[u32]) -> Arr {
        let s = &self.s;
        let idx = Arr::from_u32(ids, &[ids.len() as i32]);
        let w = self.embed.w.take_axis(&idx, 0, s);
        let sc = self.embed.scales.take_axis(&idx, 0, s);
        let bi = self.embed.biases.take_axis(&idx, 0, s);
        Arr::dequantize(&w, &sc, &bi, self.cfg.group_size as i32, self.cfg.bits as i32, Dtype::BF16, s)
            .reshape(&[1, ids.len() as i32, self.cfg.hidden as i32], s)
    }

    /// Run `ids` through the decoder at positions `cache.offset..`, updating the cache.
    /// Returns f32 logits `[vocab]` for the LAST position when `want_logits`, else None
    /// (the cache is still materialized via the returned hidden state being evaluated).
    pub fn forward(&self, ids: &[u32], cache: &mut KvCache, want_logits: bool) -> Option<Arr> {
        assert!(!ids.is_empty(), "forward on empty input");
        let s = &self.s;
        let cfg = &self.cfg;
        let l = ids.len() as i32;
        let (nh, kvh, hd) = (cfg.heads as i32, cfg.kv_heads as i32, cfg.head_dim as i32);
        let offset = cache.offset;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        let mut h = self.embed(ids);
        let attn_scale = {
            let v = cfg.attn_scales(offset, ids.len());
            let all_one = v.iter().all(|&x| x == 1.0);
            if all_one { None } else { Some(Arr::from_f32(&v, &[1, 1, l, 1]).astype(Dtype::BF16, s)) }
        };
        let mscale = if self.yarn_mscale != 1.0 { Some(Arr::scalar(self.yarn_mscale).astype(Dtype::BF16, s)) } else { None };

        for (i, layer) in self.layers.iter().enumerate() {
            let x = h.rms_norm(&layer.ln1, cfg.rms_eps, s);
            let mut q = self.proj(&x, &layer.q).reshape(&[1, l, nh, hd], s).transpose(&[0, 2, 1, 3], s);
            let mut k = self.proj(&x, &layer.k).reshape(&[1, l, kvh, hd], s).transpose(&[0, 2, 1, 3], s);
            let v = self.proj(&x, &layer.v).reshape(&[1, l, kvh, hd], s).transpose(&[0, 2, 1, 3], s);
            if let Some(m) = &mscale { q = q.mul(m, s); k = k.mul(m, s); }
            q = q.rope_freqs(hd, offset as i32, &self.freqs, s);
            k = k.rope_freqs(hd, offset as i32, &self.freqs, s);
            let (kc, vc) = cache.update(i, &k, &v, offset, s);
            if let Some(a) = &attn_scale { q = q.mul(a, s); }
            let att = Arr::sdpa(&q, &kc, &vc, scale, l > 1, s)
                .transpose(&[0, 2, 1, 3], s)
                .reshape(&[1, l, nh * hd], s);
            h = h.add(&self.proj(&att, &layer.o), s);
            let x2 = h.rms_norm(&layer.ln2, cfg.rms_eps, s);
            let gate = self.proj(&x2, &layer.gate).silu(s);
            let up = self.proj(&x2, &layer.up);
            h = h.add(&self.proj(&gate.mul(&up, s), &layer.down), s);
        }
        cache.offset = offset + ids.len();
        if want_logits {
            let last = h.slice(&[0, l - 1, 0], &[1, l, cfg.hidden as i32], s); // [1,1,H]
            let y = last.rms_norm(&self.norm, cfg.rms_eps, s);
            let logits = self.proj(&y, &self.lm_head).reshape(&[cfg.vocab as i32], s).astype(Dtype::F32, s);
            Some(logits)
        } else {
            // Evaluate the hidden state so the cache slabs written this step are materialized
            // and the graph does not keep growing across prefill chunks.
            h.eval();
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config { hidden: 4096, layers: 34, heads: 32, kv_heads: 8, head_dim: 128, intermediate: 14336, vocab: 131072, rms_eps: 1e-5,
            rope_theta: 1e6, yarn_factor: 16.0, yarn_original_max: 16384, yarn_beta_fast: 32.0, yarn_beta_slow: 1.0, yarn_mscale: 1.0,
            yarn_mscale_all_dim: 1.0, llama4_beta: 0.1, group_size: 64, bits: 4, tie_word_embeddings: false }
    }

    #[test]
    fn yarn_freqs_match_scalar_reference_formula() {
        // Independent re-derivation (mirrors upstream YarnRoPE, written separately from Config::yarn_freqs).
        let c = cfg();
        let f = c.yarn_freqs();
        assert_eq!(f.len(), 64);
        let dims = 128.0f64; let base = 1e6f64; let orig = 16384.0f64;
        let corr = |r: f64| dims * (orig / (r * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
        let low = corr(32.0).floor().max(0.0); let high = corr(1.0).ceil().min(dims - 1.0);
        for i in 0..64 {
            let fe = base.powf(2.0 * i as f64 / dims); let fi = 16.0 * fe;
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0); let mask = 1.0 - ramp;
            let want = (fi * fe) / (fi * mask + fe * (1.0 - mask));
            assert!(((f[i] as f64 - want) / want).abs() < 1e-6, "freq[{i}] {} vs {want}", f[i]);
        }
        // low-index (high frequency) pairs are unscaled, high-index pairs are interpolated by the factor
        assert!((f[0] - 1.0).abs() < 1e-6);
        assert!((f[63] as f64 / base.powf(2.0 * 63.0 / dims) - 16.0).abs() < 1e-3);
        assert_eq!(c.yarn_mscale(), 1.0);
    }

    #[test]
    fn llama4_attention_scale_is_identity_inside_original_window() {
        let c = cfg();
        assert!(c.attn_scales(0, 4096).iter().all(|&x| x == 1.0));
        let far = c.attn_scales(16384, 2);
        assert!((far[0] as f64 - (1.0 + 0.1 * (2.0f64).ln())).abs() < 1e-6);
    }
}

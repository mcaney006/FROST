//! frost-gen: the conversational generator. Ministral-3-8B-Instruct-2512 (MLX 4-bit affine)
//! running on Apple GPU through mlx-c, with a reusable KV cache, chunked prefill,
//! incremental decoding, streaming detokenization, cancellation and honest accounting.
//!
//! This is a separate model from the nomic retrieval encoder in `frost-model`; nothing here
//! relabels similarity scores as generated text.

pub mod model;
pub mod sample;
pub mod template;
pub mod tokenizer;

pub use model::{Config, KvCache, Model, ModelError};
pub use sample::{Backend, Rng, SampleParams};
pub use template::{parse_output, render, Message, Role, ToolCall};
pub use tokenizer::{ctl, Detokenizer, Tokenizer, TokenizerError};

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub const REPO: &str = "mlx-community/Ministral-3-8B-Instruct-2512-4bit";
pub const REVISION: &str = "182f003f01daa75f9de0f2c4d379722fd0bc1c61";
/// Total tokens per request (prompt + reserved output). Raised only after measured memory tests.
pub const CONTEXT_BUDGET: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub enum GenError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error("prompt of {prompt} tokens + {reserve} reserved output tokens exceeds the {budget}-token context budget")]
    ContextBudget { prompt: usize, reserve: usize, budget: usize },
    #[error("empty prompt")]
    EmptyPrompt,
    #[error("model manifest: {0}")]
    Manifest(String),
}

/// What is actually loaded, reported separately from the retrieval encoder.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Identity {
    pub repo: String,
    pub revision: String,
    pub architecture: String,
    pub quantization: String,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab: usize,
    pub context_budget: usize,
    pub backend: String,
    pub weight_bytes: usize,
    pub model_dir: String,
    /// Zig q4 layout validation summary from load ("not loaded" for `describe`).
    pub layout_check: String,
}

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new_tokens: usize,
    pub sample: SampleParams,
    /// Prompt tokens per forward pass during prefill (bounds transient attention memory).
    pub prefill_chunk: usize,
}
impl Default for GenParams {
    fn default() -> Self { GenParams { max_new_tokens: 1024, sample: SampleParams::default(), prefill_chunk: 512 } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FinishReason { Eos, MaxTokens, Cancelled }

#[derive(Debug, Clone, serde::Serialize)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub reused_prefix_tokens: usize,
    pub prefilled_tokens: usize,
    pub new_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub tokens_per_second: f64,
    pub finish: FinishReason,
    /// Backend that processed the logits on the last sampled token ("mojo-dylib", "rust-reference", "rust-argmax").
    pub sampler: &'static str,
    pub mlx_active_bytes: usize,
    pub mlx_peak_bytes: usize,
    pub kv_cache_bytes: usize,
}

pub enum Event<'a> {
    /// One prefill chunk was fed (lets callers poll governors and cancel between chunks).
    PrefillChunk { fed: usize, total: usize },
    Prefilled { tokens: usize, ms: f64 },
    Token { id: u32, text: &'a str },
}

pub struct Generator {
    model: Model,
    tok: Tokenizer,
    cache: KvCache,
    /// Token ids whose keys/values the cache currently holds (`len == cache.offset`).
    cached_ids: Vec<u32>,
    pub identity: Identity,
}

impl Generator {
    pub fn default_dir() -> PathBuf {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("Library/Application Support/FROST/models/Ministral-3-8B-Instruct-2512-4bit")
    }

    pub fn load(dir: &Path) -> Result<Generator, GenError> {
        // The manifest written by tools/fetch_ministral.sh pins the revision; refuse a mismatch.
        let manifest = dir.join("MANIFEST.json");
        if manifest.exists() {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest).map_err(|e| GenError::Manifest(e.to_string()))?)
                .map_err(|e| GenError::Manifest(e.to_string()))?;
            let rev = v["revision"].as_str().unwrap_or("");
            if rev != REVISION { return Err(GenError::Manifest(format!("revision {rev:?} != pinned {REVISION}"))); }
        }
        let model = Model::load(dir)?;
        let tok = Tokenizer::load(&dir.join("tokenizer.json"))?;
        if tok.vocab_size != model.cfg.vocab {
            return Err(GenError::Manifest(format!("tokenizer vocab {} != model vocab {}", tok.vocab_size, model.cfg.vocab)));
        }
        let cache = model.new_cache();
        let identity = Identity {
            repo: REPO.into(), revision: REVISION.into(),
            architecture: "Mistral3ForConditionalGeneration / ministral3 (text-only path; vision tower not loaded)".into(),
            quantization: format!("{}-bit affine, group size {}", model.cfg.bits, model.cfg.group_size),
            layers: model.cfg.layers, hidden: model.cfg.hidden, heads: model.cfg.heads, kv_heads: model.cfg.kv_heads,
            vocab: model.cfg.vocab, context_budget: CONTEXT_BUDGET, backend: "mlx-gpu (mlx-c)".into(),
            weight_bytes: model.weight_bytes, model_dir: dir.display().to_string(), layout_check: model.layout_check.clone(),
        };
        Ok(Generator { model, tok, cache, cached_ids: Vec::new(), identity })
    }

    /// Identity of the checkpoint in `dir` WITHOUT loading weights (config + shard headers +
    /// manifest); used by diagnostics and the installer.
    pub fn describe(dir: &Path) -> Result<Identity, GenError> {
        let cfg = Config::from_json(&dir.join("config.json"))?;
        let shards = model::Shards::open(dir)?;
        let weight_bytes: usize = shards.files.iter()
            .flat_map(|(_, st)| st.names().filter(|n| n.starts_with("language_model.")).filter_map(|n| st.info(n)).map(|i| i.nbytes()))
            .sum();
        let manifest = dir.join("MANIFEST.json");
        let revision = if manifest.exists() {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest).map_err(|e| GenError::Manifest(e.to_string()))?)
                .map_err(|e| GenError::Manifest(e.to_string()))?;
            v["revision"].as_str().unwrap_or("unknown").to_string()
        } else { "unpinned (no MANIFEST.json)".to_string() };
        Ok(Identity {
            repo: REPO.into(), revision,
            architecture: "Mistral3ForConditionalGeneration / ministral3 (text-only path; vision tower not loaded)".into(),
            quantization: format!("{}-bit affine, group size {}", cfg.bits, cfg.group_size),
            layers: cfg.layers, hidden: cfg.hidden, heads: cfg.heads, kv_heads: cfg.kv_heads, vocab: cfg.vocab,
            context_budget: CONTEXT_BUDGET, backend: "mlx-gpu (mlx-c)".into(), weight_bytes, model_dir: dir.display().to_string(), layout_check: "not loaded".into(),
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer { &self.tok }
    pub fn config(&self) -> &Config { &self.model.cfg }
    pub fn kv_cache_bytes(&self) -> usize { self.cache.nbytes() }

    /// Drop the KV cache (e.g. after a conversation switch or under memory pressure).
    pub fn reset(&mut self) { self.cache.clear(); self.cached_ids.clear(); }

    /// Generate a continuation of `prompt` (token ids). Streams tokens through `on_event`,
    /// stops on EOS, `max_new_tokens`, or when `cancel` is set. Returns the generated ids
    /// (EOS excluded) and the run's statistics. The KV cache is left consistent so the
    /// next call can reuse the longest common prompt prefix.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        p: &GenParams,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(Event),
    ) -> Result<(Vec<u32>, Stats), GenError> {
        if prompt.is_empty() { return Err(GenError::EmptyPrompt); }
        if prompt.len() + p.max_new_tokens > CONTEXT_BUDGET {
            return Err(GenError::ContextBudget { prompt: prompt.len(), reserve: p.max_new_tokens, budget: CONTEXT_BUDGET });
        }
        frost_mlx::reset_peak_memory();

        // Prefix reuse: keep the cached tokens that match, but always feed at least one token
        // so we have logits for the next position.
        let mut lcp = self.cached_ids.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        if lcp == prompt.len() { lcp -= 1; }
        self.cache.trim_to(lcp);
        self.cached_ids.truncate(lcp);

        let t0 = Instant::now();
        let rest = &prompt[lcp..];
        let chunk = p.prefill_chunk.max(1);
        let mut logits = None;
        let mut fed = 0usize;
        while fed < rest.len() {
            if cancel.load(Ordering::Relaxed) {
                return Ok((Vec::new(), self.stats(prompt.len(), lcp, fed, 0, t0.elapsed().as_secs_f64() * 1e3, 0.0, FinishReason::Cancelled)));
            }
            let end = (fed + chunk).min(rest.len());
            let last = end == rest.len();
            logits = self.model.forward(&rest[fed..end], &mut self.cache, last);
            self.cached_ids.extend_from_slice(&rest[fed..end]);
            fed = end;
            on_event(Event::PrefillChunk { fed, total: rest.len() });
        }
        let mut logits = logits.expect("prefill produced logits");
        logits.eval(); // MLX is lazy: force the prefill compute so its time is not booked to decode
        let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
        on_event(Event::Prefilled { tokens: rest.len(), ms: prefill_ms });

        let mut generated: Vec<u32> = Vec::new();
        let mut detok = self.tok.detokenizer();
        let mut rng = Rng::new(p.sample.seed);
        let mut sampler;
        let t1 = Instant::now();
        let finish = loop {
            let mut l = logits.to_vec();
            let (tok, backend) = sample::sample(&mut l, &p.sample, &generated, &mut rng);
            sampler = backend.label();
            if tok == ctl::EOS { break FinishReason::Eos; }
            generated.push(tok);
            if !self.tok.is_control(tok) {
                let text = detok.push(&self.tok.token_bytes(tok));
                on_event(Event::Token { id: tok, text: &text });
            } else {
                on_event(Event::Token { id: tok, text: "" });
            }
            if generated.len() >= p.max_new_tokens { break FinishReason::MaxTokens; }
            if cancel.load(Ordering::Relaxed) { break FinishReason::Cancelled; }
            // feed the token; its keys/values extend the cache
            logits = self.model.forward(&[tok], &mut self.cache, true).expect("decode produced logits");
            self.cached_ids.push(tok);
        };
        let tail = detok.finish();
        if !tail.is_empty() { on_event(Event::Token { id: u32::MAX, text: &tail }); }
        let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        let mut stats = self.stats(prompt.len(), lcp, rest.len(), generated.len(), prefill_ms, decode_ms, finish);
        stats.sampler = sampler;
        Ok((generated, stats))
    }

    #[allow(clippy::too_many_arguments)]
    fn stats(&self, prompt: usize, lcp: usize, prefilled: usize, new: usize, prefill_ms: f64, decode_ms: f64, finish: FinishReason) -> Stats {
        let mem = frost_mlx::memory();
        Stats {
            prompt_tokens: prompt, reused_prefix_tokens: lcp, prefilled_tokens: prefilled, new_tokens: new,
            prefill_ms, decode_ms,
            tokens_per_second: if decode_ms > 0.0 { new as f64 / (decode_ms / 1e3) } else { 0.0 },
            finish, sampler: "none",
            mlx_active_bytes: mem.active, mlx_peak_bytes: mem.peak, kv_cache_bytes: self.cache.nbytes(),
        }
    }
}

/// The FROST default system prompt (the generator never sees the checkpoint's "Le Chat" default).
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are FROST, a local coding assistant running entirely on this Mac. \
Answer directly and concisely. When you show code, use fenced code blocks with a language tag. \
If you are unsure, say so instead of guessing. You cannot browse the web or run commands yourself; \
FROST executes tools only after the user approves them.";

#[cfg(test)]
mod acceptance {
    //! Mandatory acceptance tests. They need the pinned checkpoint and FAIL (never skip) when
    //! it is missing; `bootstrap.sh` runs them with `cargo test -p frost-gen -- --ignored`.
    use super::*;
    use std::sync::Mutex;

    /// One model per process at a time: two concurrent loads would exceed 16 GB and swap.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn load() -> (Generator, std::sync::MutexGuard<'static, ()>) {
        let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = Generator::default_dir();
        (Generator::load(&dir).unwrap_or_else(|e| panic!("REQUIRED asset missing/invalid at {}: {e}", dir.display())), guard)
    }

    #[allow(dead_code)]
    fn load_only() -> Generator {
        let dir = Generator::default_dir();
        Generator::load(&dir).unwrap_or_else(|e| panic!("REQUIRED asset missing/invalid at {}: {e}", dir.display()))
    }

    #[test]
    #[ignore = "requires the 5.6 GB pinned checkpoint; run explicitly"]
    fn generates_coherent_multi_turn_text_and_reuses_prefix() {
        let (mut g, _serial) = load();
        let tok = g.tokenizer();
        let mut msgs = vec![Message::user("Reply with exactly one word: the capital of France.")];
        let prompt = render(tok, DEFAULT_SYSTEM_PROMPT, None, &msgs).unwrap();
        let p = GenParams { max_new_tokens: 16, sample: SampleParams { temperature: 0.0, ..Default::default() }, ..Default::default() };
        let cancel = AtomicBool::new(false);
        let mut out = String::new();
        let (ids, st) = g.generate(&prompt, &p, &cancel, &mut |e| if let Event::Token { text, .. } = e { out.push_str(text) }).unwrap();
        eprintln!("turn1: {out:?} stats={st:?}");
        assert!(out.to_lowercase().contains("paris"), "expected Paris, got {out:?}");
        assert!(matches!(st.finish, FinishReason::Eos | FinishReason::MaxTokens));
        assert!(st.tokens_per_second > 1.0);
        // second turn: the whole first exchange must be reused from the cache
        msgs.push(Message::assistant(g.tokenizer().decode(&ids)));
        msgs.push(Message::user("And of Italy? One word."));
        let prompt2 = render(g.tokenizer(), DEFAULT_SYSTEM_PROMPT, None, &msgs).unwrap();
        let mut out2 = String::new();
        let (_, st2) = g.generate(&prompt2, &p, &cancel, &mut |e| if let Event::Token { text, .. } = e { out2.push_str(text) }).unwrap();
        eprintln!("turn2: {out2:?} stats={st2:?}");
        assert!(out2.to_lowercase().contains("rome"), "expected Rome, got {out2:?}");
        assert!(st2.reused_prefix_tokens >= prompt.len() - 1, "prefix reuse: {} of {}", st2.reused_prefix_tokens, prompt.len());
    }

    #[test]
    #[ignore = "requires the pinned checkpoint"]
    fn cancellation_is_safe_and_the_next_request_works() {
        let (mut g, _serial) = load();
        let prompt = render(g.tokenizer(), DEFAULT_SYSTEM_PROMPT, None, &[Message::user("Count from 1 to 200, one number per line.")]).unwrap();
        let p = GenParams { max_new_tokens: 200, sample: SampleParams { temperature: 0.0, ..Default::default() }, ..Default::default() };
        let cancel = AtomicBool::new(false);
        let mut n = 0;
        let (_, st) = g.generate(&prompt, &p, &cancel, &mut |e| if let Event::Token { .. } = e { n += 1; if n == 5 { cancel.store(true, Ordering::Relaxed); } }).unwrap();
        assert_eq!(st.finish, FinishReason::Cancelled);
        assert!(st.new_tokens >= 5 && st.new_tokens <= 6, "{}", st.new_tokens);
        let cancel = AtomicBool::new(false);
        let prompt2 = render(g.tokenizer(), DEFAULT_SYSTEM_PROMPT, None, &[Message::user("Say hello.")]).unwrap();
        let mut out = String::new();
        let (_, st2) = g.generate(&prompt2, &p, &cancel, &mut |e| if let Event::Token { text, .. } = e { out.push_str(text) }).unwrap();
        assert!(st2.new_tokens > 0 && !out.trim().is_empty(), "second request after cancel produced {out:?}");
    }

    #[test]
    #[ignore = "requires the pinned checkpoint"]
    fn refuses_prompts_over_the_context_budget_instead_of_truncating() {
        let (mut g, _serial) = load();
        let ids = vec![ctl::BOS; CONTEXT_BUDGET - 10];
        let p = GenParams { max_new_tokens: 64, ..Default::default() };
        let r = g.generate(&ids, &p, &AtomicBool::new(false), &mut |_| {});
        assert!(matches!(r, Err(GenError::ContextBudget { .. })));
    }
}

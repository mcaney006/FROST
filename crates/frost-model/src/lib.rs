//! frost-model: real model-backed embeddings for FROST.
//! Full-depth nomic-embed-text-v1.5 encoder on MLX (GPU), the reference/fallback
//! ranker per the build contract. Progressive heads layer on top later.
pub use frost_mlx as mlx;
pub mod model;
pub mod tokenizer;
pub mod reference;

use anyhow::Result;
use std::path::{Path, PathBuf};

pub const REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

/// nomic task prefixes (part of the input text, per the model card).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task { SearchQuery, SearchDocument, Classification, Clustering }

impl Task {
    pub fn prefix(&self) -> &'static str {
        match self {
            Task::SearchQuery => "search_query: ",
            Task::SearchDocument => "search_document: ",
            Task::Classification => "classification: ",
            Task::Clustering => "clustering: ",
        }
    }
}

pub struct Embedder {
    model: model::Model,
    tok: tokenizer::Tokenizer,
    pub max_tokens: usize,
}

impl Embedder {
    pub fn default_dir() -> PathBuf {
        PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Application Support/FROST/models/nomic-embed-text-v1.5")
    }

    pub fn load(dir: &Path) -> Result<Embedder> {
        let model = model::Model::load(dir, REVISION)?;
        let tok = tokenizer::Tokenizer::load(&dir.join("vocab.txt"))?;
        Ok(Embedder { model, tok, max_tokens: 512 })
    }

    pub fn dim(&self) -> usize { self.model.cfg.hidden }
    pub fn fingerprint(&self) -> &str { &self.model.fingerprint }

    /// Encode text for a task into an L2-normalized embedding.
    pub fn encode(&self, text: &str, task: Task) -> Vec<f32> {
        let full = format!("{}{}", task.prefix(), text);
        let ids = self.tok.encode(&full, self.max_tokens);
        model::l2_normalize(self.model.forward(&ids))
    }

    /// Cosine scores of `query` (SearchQuery) vs each doc (SearchDocument).
    pub fn rank(&self, query: &str, docs: &[String]) -> Vec<f32> {
        let q = self.encode(query, Task::SearchQuery);
        docs.iter()
            .map(|d| {
                let e = self.encode(d, Task::SearchDocument);
                q.iter().zip(&e).map(|(a, b)| a * b).sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod acceptance {
    //! Mandatory, weight-backed tests: they FAIL when the pinned encoder is missing and are run
    //! explicitly (`cargo test -p frost-model -- --ignored`) by bootstrap/verification.
    use super::*;

    fn load() -> Embedder {
        let dir = Embedder::default_dir();
        Embedder::load(&dir).unwrap_or_else(|e| panic!("REQUIRED encoder missing/invalid at {}: {e:#}", dir.display()))
    }

    fn cos_f32(a: &[f32], b: &[f32]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        d / (na * nb)
    }

    #[test]
    #[ignore = "requires the pinned nomic weights; run explicitly"]
    fn real_model_behavioral_ranking() {
        let emb = load();
        assert_eq!(emb.dim(), 768);
        eprintln!("fingerprint = {}", emb.fingerprint());
        // determinism + normalization
        let a = emb.encode("hello world", Task::SearchQuery);
        let b = emb.encode("hello world", Task::SearchQuery);
        assert_eq!(a, b, "deterministic");
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "L2 norm {norm}");
        // relevant doc must clearly outrank the irrelevant one
        let scores = emb.rank("what is a dog", &[
            "a dog is a domesticated carnivore of the family Canidae".into(),
            "the federal reserve raised interest rates this quarter".into(),
        ]);
        eprintln!("scores = {scores:?}");
        assert!(scores[0] > scores[1] + 0.2, "relevant must win clearly: {scores:?}");
    }

    #[test]
    #[ignore = "requires the pinned nomic weights; run explicitly"]
    fn mlx_matches_scalar_reference() {
        // Independent scalar (f64 CPU) reference vs MLX (GPU) forward — parity.
        let emb = load();
        let dir = Embedder::default_dir();
        let r = reference::Ref::load(&dir).unwrap();
        let ids = emb.tok.encode("search_document: a dog is a domesticated canine", 64);
        let mlx = model::l2_normalize(emb.model.forward(&ids));
        let scal: Vec<f32> = r.embed(&ids).iter().map(|x| *x as f32).collect();
        let c = cos_f32(&mlx, &scal);
        eprintln!("scalar/MLX parity cos = {c:.6}");
        assert!(c > 0.9999, "MLX must match scalar reference: cos={c}");
        // partial-forward parity: the full-depth partial path is the ordinary forward
        let full = emb.model.forward(&ids);
        let upto = emb.model.forward_upto(&ids, emb.model.cfg.layers);
        assert_eq!(full, upto, "forward_upto(all layers) must equal forward");
        let half = model::l2_normalize(emb.model.forward_upto(&ids, emb.model.cfg.layers / 2));
        assert!(cos_f32(&half, &mlx) < 0.9999, "half-depth output must differ from full depth");
    }
}

//! The one shared FROST engine. GUI, CLI, and API all call this — behavior
//! cannot diverge between surfaces. Owns the model, the Mojo scoring kernel,
//! and an exact-decision cache keyed by a full fingerprint.

use crate::kernel::MojoKernel;
use frost_core::{
    fnv1a64, Calibration, Decision, DeferReason, Exit, Request, Scored, ThermalState, Trace,
};
use frost_model::{Embedder, Task};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

pub struct Engine {
    embedder: Embedder,
    kernel: Option<MojoKernel>,
    cache: Mutex<HashMap<u64, (Vec<Scored>, Option<String>)>>,
    calibration: Calibration,
    device: String,
    thermal_override: Mutex<Option<ThermalState>>,
}

impl Engine {
    /// Load the engine from a model directory, spawning the Mojo kernel if its
    /// compiled helper is present (falls back to the Zig exact scorer otherwise).
    pub fn load(model_dir: &Path) -> anyhow::Result<Engine> {
        let embedder = Embedder::load(model_dir)?;
        let kpath = MojoKernel::default_path();
        let kernel = MojoKernel::spawn(&kpath).ok();
        Ok(Engine {
            embedder,
            kernel,
            cache: Mutex::new(HashMap::new()),
            // Full-depth reference ranker is enabled with a conservative margin gate.
            calibration: Calibration { exit: Exit::Full, width: 768, min_margin: 0.05, enabled: true },
            device: "gpu".into(),
            thermal_override: Mutex::new(None),
        })
    }

    pub fn fingerprint(&self) -> &str { self.embedder.fingerprint() }
    pub fn scorer(&self) -> &str { if self.kernel.is_some() { "mojo-ipc" } else { "zig-fp32" } }

    /// Simulate a thermal state (for tests / diagnostics). None = read the OS.
    pub fn set_thermal_override(&self, t: Option<ThermalState>) { *self.thermal_override.lock().unwrap() = t; }
    fn thermal(&self) -> ThermalState {
        self.thermal_override.lock().unwrap().unwrap_or_else(frost_platform::thermal_state)
    }

    /// Cache key: model fp + mode + the ordered eligible (id|text|permission) set.
    fn cache_key(&self, req: &Request) -> u64 {
        let mut s = format!("{}|{:?}|", self.embedder.fingerprint(), req.mode);
        let mut items: Vec<String> = req
            .eligible()
            .iter()
            .map(|c| format!("{}\u{1}{}\u{1}{}", c.id, c.text, c.permission.clone().unwrap_or_default()))
            .collect();
        items.sort();
        s.push_str(&format!("state={}|", req.state));
        for it in items { s.push_str(&it); s.push('\u{2}'); }
        fnv1a64(s.as_bytes())
    }

    /// Score `query` embedding against each candidate embedding. Uses the Mojo
    /// kernel when present (real compiled-Mojo path); else the Zig fp32 dot.
    fn score(&self, q: &[f32], cands: &[Vec<f32>]) -> Vec<f32> {
        if let Some(k) = &self.kernel {
            if let Ok(s) = k.score(q, cands) {
                return s;
            }
        }
        // fallback: exact fp32 dot via the Zig kernel (vectors are L2-normalized).
        cands.iter().map(|c| frost_index::dot_f32(q, c)).collect()
    }

    pub fn rank(&self, req: &Request) -> Decision {
        let eligible = req.eligible();
        if eligible.is_empty() {
            return Decision::Deferred { reason: DeferReason::EmptyBank, trace: self.base_trace(false) };
        }

        let thermal = self.thermal();
        let key = self.cache_key(req);
        if let Some((ranking, pick)) = self.cache.lock().unwrap().get(&key).cloned() {
            // cache hits do no new neural work, so they are allowed under any thermal state.
            let mut tr = self.base_trace(true);
            tr.thermal = Some(format!("{thermal:?}"));
            tr.retrieved = ranking.len();
            tr.reranked = ranking.len();
            tr.notes.push("exact-cache hit: encoder skipped".into());
            return Decision::Ranked { ranking, pick, trace: tr };
        }

        // Serious/critical: stop admitting UNCACHED neural work; defer visibly.
        if matches!(thermal, ThermalState::Serious | ThermalState::Critical) {
            let mut tr = self.base_trace(false);
            tr.thermal = Some(format!("{thermal:?}"));
            tr.notes.push("deferred: thermal state does not admit new neural work".into());
            return Decision::Deferred { reason: DeferReason::ThermalCritical, trace: tr };
        }

        let mut tr = self.base_trace(false);
        tr.thermal = Some(format!("{thermal:?}"));
        let t0 = Instant::now();
        let q = self.embedder.encode(&req.state, Task::SearchQuery);
        let cand_vecs: Vec<Vec<f32>> = eligible
            .iter()
            .map(|c| self.embedder.encode(&c.text, Task::SearchDocument))
            .collect();
        tr.stage_ms.push(("encode".into(), t0.elapsed().as_secs_f64() * 1e3));

        let t1 = Instant::now();
        let scores = self.score(&q, &cand_vecs);
        tr.stage_ms.push(("score".into(), t1.elapsed().as_secs_f64() * 1e3));

        let mut ranking: Vec<Scored> = eligible
            .iter()
            .zip(&scores)
            .map(|(c, &s)| Scored { id: c.id.clone(), raw_score: s, calibrated: None })
            .collect();
        // stable sort desc by raw score; deterministic tie-break by id
        ranking.sort_by(|a, b| {
            b.raw_score.partial_cmp(&a.raw_score).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        tr.retrieved = eligible.len();
        tr.reranked = eligible.len();
        tr.notes.push(format!("scorer={}", self.scorer()));

        let pick = self.calibration.gate_pick(&ranking);
        if pick.is_none() {
            tr.notes.push("abstained on pick: top margin below calibrated threshold".into());
        }

        self.cache.lock().unwrap().insert(key, (ranking.clone(), pick.clone()));
        Decision::Ranked { ranking, pick, trace: tr }
    }

    fn base_trace(&self, cache_hit: bool) -> Trace {
        Trace {
            exit: Some("full".into()),
            width: Some(self.embedder.dim() as u32),
            retrieved: 0,
            reranked: 0,
            device: self.device.clone(),
            cache_hit,
            stage_ms: vec![],
            model_fingerprint: self.embedder.fingerprint().to_string(),
            head_fingerprint: None,
            thermal: None,
            notes: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frost_core::{Candidate, Mode};

    fn req(state: &str, acts: &[(&str, &str)]) -> Request {
        Request {
            state: state.into(),
            candidates: acts.iter().map(|(id, t)| Candidate { id: (*id).into(), text: (*t).into(), permission: None }).collect(),
            mode: Mode::Quiet,
            allowed: None,
        }
    }

    #[test]
    fn ranks_relevant_action_first_and_caches() {
        let dir = Embedder::default_dir();
        if !dir.join("model.safetensors").exists() { eprintln!("SKIP: no weights"); return; }
        let eng = Engine::load(&dir).expect("engine");
        let r = req("the user wants to play music", &[
            ("play", "start playing a song from the library"),
            ("delete", "permanently erase all backup files"),
            ("email", "compose and send an email to a colleague"),
        ]);
        let d = eng.rank(&r);
        match d {
            Decision::Ranked { ranking, trace, .. } => {
                eprintln!("scorer={} top={} score={:.3}", eng.scorer(), ranking[0].id, ranking[0].raw_score);
                assert_eq!(ranking[0].id, "play", "music intent should pick 'play'");
                assert!(!trace.cache_hit);
                assert_eq!(trace.device, "gpu");
            }
            _ => panic!("expected ranked"),
        }
        // second identical request -> cache hit, encoder skipped
        let d2 = eng.rank(&r);
        if let Decision::Ranked { trace, .. } = d2 {
            assert!(trace.cache_hit, "identical request must hit cache");
        } else { panic!(); }
    }

    #[test]
    fn empty_bank_defers() {
        let dir = Embedder::default_dir();
        if !dir.join("model.safetensors").exists() { return; }
        let eng = Engine::load(&dir).unwrap();
        let d = eng.rank(&req("anything", &[]));
        assert!(matches!(d, Decision::Deferred { reason: DeferReason::EmptyBank, .. }));
    }

    #[test]
    fn changed_candidate_set_is_not_reused() {
        let dir = Embedder::default_dir();
        if !dir.join("model.safetensors").exists() { return; }
        let eng = Engine::load(&dir).unwrap();
        let a = req("state x", &[("a", "alpha action")]);
        let _ = eng.rank(&a);
        let b = req("state x", &[("a", "alpha action"), ("b", "beta action")]);
        if let Decision::Ranked { trace, .. } = eng.rank(&b) {
            assert!(!trace.cache_hit, "different candidate set must NOT reuse cache");
        } else { panic!(); }
    }
}

#[cfg(test)]
mod thermal_tests {
    use super::*;
    use frost_core::{Candidate, Mode, ThermalState};
    fn r() -> Request {
        Request { state: "play music".into(),
            candidates: vec![Candidate{id:"p".into(),text:"play a song".into(),permission:None},
                             Candidate{id:"d".into(),text:"delete files".into(),permission:None}],
            mode: Mode::Quiet, allowed: None }
    }
    #[test]
    fn serious_defers_uncached_but_cache_still_serves() {
        let dir = Embedder::default_dir();
        if !dir.join("model.safetensors").exists() { return; }
        let eng = Engine::load(&dir).unwrap();
        // warm the cache under nominal
        eng.set_thermal_override(Some(ThermalState::Nominal));
        assert!(matches!(eng.rank(&r()), Decision::Ranked{..}));
        // simulate serious: a NEW request defers; the cached one still serves
        eng.set_thermal_override(Some(ThermalState::Serious));
        let novel = Request { state: "totally different novel state".into(), ..r() };
        assert!(matches!(eng.rank(&novel), Decision::Deferred{ reason: DeferReason::ThermalCritical, .. }),
            "uncached work must defer under serious");
        if let Decision::Ranked { trace, .. } = eng.rank(&r()) {
            assert!(trace.cache_hit, "cached decision still served under serious");
        } else { panic!("cached request should still serve"); }
        // critical also defers
        eng.set_thermal_override(Some(ThermalState::Critical));
        assert!(matches!(eng.rank(&Request{state:"another new one".into(),..r()}), Decision::Deferred{..}));
    }
}

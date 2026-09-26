//! frost-core: request/decision types, candidate filtering, policies, calibration.
//!
//! No I/O, no model, no FFI here — pure logic that every surface (CLI, service,
//! GUI) shares so behavior can't diverge between them.

use serde::{Deserialize, Serialize};

/// Thermal scheduling posture. Distinct from an actual temperature measurement:
/// these are application scheduling policies driven by OS thermal *state*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

/// User-selected performance policy. Default is Quiet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Quiet,
    Balanced,
    Performance,
}

impl Mode {
    /// Interactive input token budget for the mode.
    pub fn input_token_budget(&self) -> usize {
        match self {
            Mode::Quiet => 512,
            Mode::Balanced => 1024,
            Mode::Performance => 2048,
        }
    }
    /// Hard interactive cap until a larger-context profile is separately tested.
    pub const HARD_TOKEN_CAP: usize = 2048;
}

/// Which encoder exit produced a representation. Depths derive from the real
/// architecture at load time; these are the logical tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Exit {
    Shallow, // ~1/3 depth
    Mid,     // ~2/3 depth
    Full,    // full depth (reference / fallback)
}

/// Nested representation width (Matryoshka-style prefixes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Width {
    W64 = 64,
    W128 = 128,
    W256 = 256,
    Full = 0, // full model width, resolved at load time
}

/// A candidate action the engine may rank. `id` must be unique within a bank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub text: String,
    /// Optional permission tag; changing permissions must invalidate cache.
    #[serde(default)]
    pub permission: Option<String>,
}

/// A decision request: a state plus the allowed candidate set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub state: String,
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub mode: Mode,
    /// If set, only candidate ids in this list are eligible.
    #[serde(default)]
    pub allowed: Option<Vec<String>>,
}

impl Request {
    /// Apply the allowed-id filter, returning the eligible candidates.
    /// A missing filter means all candidates are eligible.
    pub fn eligible(&self) -> Vec<&Candidate> {
        match &self.allowed {
            None => self.candidates.iter().collect(),
            Some(ids) => self
                .candidates
                .iter()
                .filter(|c| ids.iter().any(|a| a == &c.id))
                .collect(),
        }
    }
}

/// One scored candidate in a ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scored {
    pub id: String,
    /// Raw similarity (cosine), reported separately from calibrated confidence.
    pub raw_score: f32,
    /// Calibrated probability, only present when calibration applies to this
    /// exit/width and candidate-set domain.
    pub calibrated: Option<f32>,
}

/// Why the engine deferred or abstained instead of returning a confident pick.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeferReason {
    ThermalCritical,
    MemoryPressure,
    QueueOverflow,
    InsufficientCalibration,
    CloseAlternatives,
    IncompatibleBank,
    EmptyBank,
}

/// The outcome of a decision. Either a ranking with an optional top pick, or a
/// deferral/abstention with a reason. A thermal limit must never silently turn
/// an uncertain answer into a confident one — it produces `Deferred`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Decision {
    Ranked {
        ranking: Vec<Scored>,
        /// Top pick id, only when it clears the confidence/abstention gate.
        pick: Option<String>,
        trace: Trace,
    },
    Deferred {
        reason: DeferReason,
        trace: Trace,
    },
}

/// Inspectable execution details for a decision.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trace {
    pub exit: Option<String>,
    pub width: Option<u32>,
    pub retrieved: usize,
    pub reranked: usize,
    pub device: String,
    pub cache_hit: bool,
    pub stage_ms: Vec<(String, f64)>,
    pub model_fingerprint: String,
    pub head_fingerprint: Option<String>,
    pub thermal: Option<String>,
    pub notes: Vec<String>,
}

/// Calibration for one (exit, width) cell. `min_margin` is the score-margin a
/// top pick must clear before we return it as a confident pick; below it we
/// return a ranking with `pick=None` (abstain on the pick, still show scores).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Calibration {
    pub exit: Exit,
    pub width: u32,
    /// Required margin between top-1 and top-2 raw scores.
    pub min_margin: f32,
    /// Whether this cell met its recorded criterion and may be enabled.
    pub enabled: bool,
}

impl Calibration {
    /// Decide the pick given a *descending-score* ranking. Returns the top id
    /// only if the margin gate is cleared; otherwise None (abstain on pick).
    pub fn gate_pick(&self, ranking: &[Scored]) -> Option<String> {
        match ranking {
            [] => None,
            [only] => Some(only.id.clone()),
            [top, second, ..] => {
                if top.raw_score - second.raw_score >= self.min_margin {
                    Some(top.id.clone())
                } else {
                    None
                }
            }
        }
    }
}

/// Everything that must match for a cached decision to be reusable. Any change
/// to any field invalidates the entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub backbone: String,
    pub tokenizer: String,
    pub prefixes: String,
    pub pooling: String,
    pub exit: String,
    pub width: u32,
    pub quantizer: String,
    pub head: Option<String>,
    pub bank_generation: u64,
    /// Hash over the *allowed* candidate id set — a decision must never be
    /// reused across different allowed sets.
    pub allowed_set_hash: u64,
    pub permissions_hash: u64,
}

/// FNV-1a 64-bit, used for deterministic small hashes (allowed set, permissions).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hash a set of candidate ids order-independently (XOR of per-id hashes).
pub fn hash_id_set<'a, I: IntoIterator<Item = &'a str>>(ids: I) -> u64 {
    let mut acc: u64 = 0;
    for id in ids {
        acc ^= fnv1a64(id.as_bytes());
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str) -> Candidate {
        Candidate { id: id.into(), text: format!("do {id}"), permission: None }
    }

    #[test]
    fn eligible_respects_allowed_filter() {
        let req = Request {
            state: "s".into(),
            candidates: vec![cand("a"), cand("b"), cand("c")],
            mode: Mode::Quiet,
            allowed: Some(vec!["a".into(), "c".into()]),
        };
        let e: Vec<_> = req.eligible().iter().map(|c| c.id.clone()).collect();
        assert_eq!(e, vec!["a".to_string(), "c".to_string()]);
    }

    #[test]
    fn allowed_set_hash_is_order_independent_and_distinct() {
        let h1 = hash_id_set(["a", "b", "c"]);
        let h2 = hash_id_set(["c", "b", "a"]);
        let h3 = hash_id_set(["a", "b"]);
        assert_eq!(h1, h2, "order must not change the hash");
        assert_ne!(h1, h3, "different sets must differ (cache-safety invariant)");
    }

    #[test]
    fn margin_gate_abstains_on_close_call() {
        let cal = Calibration { exit: Exit::Full, width: 256, min_margin: 0.05, enabled: true };
        let close = vec![
            Scored { id: "a".into(), raw_score: 0.90, calibrated: None },
            Scored { id: "b".into(), raw_score: 0.88, calibrated: None },
        ];
        assert_eq!(cal.gate_pick(&close), None, "0.02 margin < 0.05 must abstain");
        let clear = vec![
            Scored { id: "a".into(), raw_score: 0.90, calibrated: None },
            Scored { id: "b".into(), raw_score: 0.10, calibrated: None },
        ];
        assert_eq!(cal.gate_pick(&clear), Some("a".into()));
    }

    #[test]
    fn quiet_budget_and_cap() {
        assert_eq!(Mode::Quiet.input_token_budget(), 512);
        assert!(Mode::Performance.input_token_budget() <= Mode::HARD_TOKEN_CAP);
    }
}

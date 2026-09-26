//! Bounded, agent-authored synthetic routing dataset + a real train→save→
//! reload→rank cycle over the frozen encoder. Train and eval use DIFFERENT
//! phrasing templates per intent (family split — no paraphrase leakage).

use crate::{loss_and_grad, sgd_step, Batch, Head, Mat};
use frost_core::fnv1a64;
use frost_model::{Embedder, Task};
use serde::Serialize;
use std::path::Path;

/// Shared action bank (id, text). One correct action per intent.
pub const ACTIONS: &[(&str, &str)] = &[
    ("play_music", "start playing a song from the music library"),
    ("delete_files", "permanently delete the selected files"),
    ("send_email", "compose and send an email message"),
    ("set_brightness", "adjust the display screen brightness"),
    ("set_alarm", "set an alarm clock for a specific time"),
    ("search_web", "search the web for information"),
    ("take_screenshot", "capture a screenshot of the screen"),
    ("lock_screen", "lock the computer screen"),
];

/// Train templates (2 phrasings per intent). Distinct families from eval.
const TRAIN: &[(&str, &str)] = &[
    ("play_music", "put on some music please"),
    ("play_music", "i want to hear a song"),
    ("delete_files", "get rid of these files for good"),
    ("delete_files", "wipe those documents permanently"),
    ("send_email", "shoot a message over to my coworker"),
    ("send_email", "draft a note and email it out"),
    ("set_brightness", "the screen is too dim, brighten it"),
    ("set_brightness", "turn up the display brightness"),
    ("set_alarm", "wake me up at seven tomorrow"),
    ("set_alarm", "set a timer alarm for the morning"),
    ("search_web", "look this up online for me"),
    ("search_web", "find information on the internet"),
    ("take_screenshot", "grab a picture of what's on screen"),
    ("take_screenshot", "capture whatever is displayed right now"),
    ("lock_screen", "secure my computer, i'm stepping away"),
    ("lock_screen", "lock things down before i leave"),
];

/// Held-out eval templates (different phrasings again).
const EVAL: &[(&str, &str)] = &[
    ("play_music", "can you start my playlist"),
    ("play_music", "begin audio playback of a track"),
    ("delete_files", "erase all of these documents"),
    ("delete_files", "remove those files entirely"),
    ("send_email", "send an email to the team"),
    ("send_email", "write and dispatch a mail"),
    ("set_brightness", "make the display brighter"),
    ("set_brightness", "change how bright the monitor is"),
    ("set_alarm", "schedule an alarm for 6am"),
    ("set_alarm", "remind me with an alarm at dawn"),
    ("search_web", "google something for me"),
    ("search_web", "run an internet search query"),
    ("take_screenshot", "screenshot the current window"),
    ("take_screenshot", "save an image of the display"),
    ("lock_screen", "lock the screen now"),
    ("lock_screen", "engage the screen lock"),
];

#[derive(Serialize)]
pub struct Report {
    pub seed: u64,
    pub epochs: usize,
    pub lr: f32,
    pub temp: f32,
    pub width: usize,
    pub n_actions: usize,
    pub n_train: usize,
    pub n_eval: usize,
    pub data_hash: u64,
    pub loss_start: f32,
    pub loss_final: f32,
    pub baseline_top1: f32,   // raw full-depth embedding cosine
    pub head_top1: f32,       // trained projection head cosine
    pub reload_matches: bool, // reloaded checkpoint reproduces the same ranking
}

fn action_index(id: &str) -> usize { ACTIONS.iter().position(|(a, _)| *a == id).unwrap() }

fn data_hash() -> u64 {
    let mut s = String::new();
    for set in [ACTIONS, TRAIN, EVAL] { for (a, b) in set { s.push_str(a); s.push('|'); s.push_str(b); s.push('\n'); } }
    fnv1a64(s.as_bytes())
}

/// Run the full cycle. Returns a report and writes the checkpoint.
pub fn train_and_eval(emb: &Embedder, ckpt: &Path, seed: u64, epochs: usize, lr: f32, temp: f32) -> anyhow::Result<Report> {
    let d = emb.dim();
    // embed action bank + train + eval (frozen encoder)
    let act_emb: Vec<Vec<f32>> = ACTIONS.iter().map(|(_, t)| emb.encode(t, Task::SearchDocument)).collect();
    let train_emb: Vec<(usize, Vec<f32>)> = TRAIN.iter()
        .map(|(id, t)| (action_index(id), emb.encode(t, Task::SearchQuery))).collect();
    let eval_emb: Vec<(usize, Vec<f32>)> = EVAL.iter()
        .map(|(id, t)| (action_index(id), emb.encode(t, Task::SearchQuery))).collect();

    // Action-bank embedding matrix [M,D]
    let m = ACTIONS.len();
    let mut xa = Mat::zeros(m, d);
    for (i, e) in act_emb.iter().enumerate() { for k in 0..d { xa.set(i, k, e[k]); } }

    // Build two batches (one per training template) so each batch has distinct
    // positives (clean diagonal): batch t uses TRAIN[2*i + t].
    let build_batch = |t: usize| -> Batch {
        let rows: Vec<&(usize, Vec<f32>)> = (0..m).map(|i| &train_emb[2 * i + t]).collect();
        let mut xs = Mat::zeros(m, d);
        for (i, (_lbl, e)) in rows.iter().enumerate() { for k in 0..d { xs.set(i, k, e[k]); } }
        // reorder xa so diagonal j==i is the positive: here positives are ACTIONS[i]
        Batch { xs, xa: xa.clone() }
    };
    let batches = [build_batch(0), build_batch(1)];

    let mut head = Head::init(d, 256, temp, seed);
    let loss_start = 0.5 * (loss_and_grad(&head, &batches[0]).0 + loss_and_grad(&head, &batches[1]).0);
    let mut loss_final = loss_start;
    for _ in 0..epochs {
        for b in &batches {
            let (l, dws, dwa) = loss_and_grad(&head, b);
            sgd_step(&mut head, &dws, &dwa, lr);
            loss_final = l;
        }
    }

    // save + reload (optimizer-independent inference checkpoint)
    head.save(ckpt)?;
    let reloaded = Head::load(ckpt)?;

    // evaluate top-1 routing accuracy on held-out eval set
    let proj_actions_head: Vec<Vec<f32>> = act_emb.iter().map(|e| reloaded.project_action(e)).collect();
    let top1 = |score: &dyn Fn(&[f32]) -> Vec<f32>| -> f32 {
        let mut correct = 0;
        for (lbl, e) in &eval_emb {
            let scores = score(e);
            let pick = scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pick == *lbl { correct += 1; }
        }
        correct as f32 / eval_emb.len() as f32
    };
    let baseline = top1(&|e| act_emb.iter().map(|a| dot(e, a)).collect());
    let head_acc = top1(&|e| { let q = reloaded.project_state(e); proj_actions_head.iter().map(|a| dot(&q, a)).collect() });

    // reload determinism: reloaded head reproduces same pick on first eval item as the in-memory head
    let q0 = head.project_state(&eval_emb[0].1);
    let q0r = reloaded.project_state(&eval_emb[0].1);
    let reload_matches = q0.iter().zip(&q0r).all(|(a, b)| (a - b).abs() < 1e-6);

    Ok(Report {
        seed, epochs, lr, temp, width: 256, n_actions: m, n_train: TRAIN.len(), n_eval: EVAL.len(),
        data_hash: data_hash(), loss_start, loss_final, baseline_top1: baseline, head_top1: head_acc, reload_matches,
    })
}

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_train_save_reload_rank_cycle() {
        let dir = Embedder::default_dir();
        if !dir.join("model.safetensors").exists() { eprintln!("SKIP: no weights"); return; }
        let emb = Embedder::load(&dir).unwrap();
        let ckpt = std::env::temp_dir().join("frost_head_test.json");
        let rep = train_and_eval(&emb, &ckpt, 1234, 150, 0.05, 0.05).unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&rep).unwrap());
        assert!(rep.loss_final < rep.loss_start, "training must reduce loss");
        assert!(rep.reload_matches, "reloaded checkpoint must reproduce projections");
        assert!(rep.head_top1 >= 0.75, "trained head should route held-out states well: {}", rep.head_top1);
        assert!(ckpt.exists(), "checkpoint file must be written");
    }
}

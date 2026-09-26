//! frost — CLI over the shared FROST engine (same engine as the GUI/API).
use anyhow::{bail, Result};
use frost_core::{Candidate, Decision, Mode, Request};
use frost_model::Embedder;
use frost_service::Engine;
use std::path::PathBuf;
use std::time::Instant;

fn model_dir() -> PathBuf { Embedder::default_dir() }

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "diagnose" => diagnose(),
        "models" => models(),
        "rank" => rank(&args[1..]),
        "bench" => bench(&args[1..]),
        "train" => train(&args[1..]),
        "help" | "-h" | "--help" => { print_help(); Ok(()) }
        other => { eprintln!("unknown command: {other}\n"); print_help(); std::process::exit(2); }
    }
}

fn print_help() {
    println!(r#"frost — offline thermally-aware decision engine

USAGE:
  frost diagnose                     check components (model, GPU, Zig, Mojo)
  frost models                       show model provenance + fingerprint
  frost rank --state S --action id=text [--action ...] [--json]
  frost bench [--n N]                encode-latency micro-benchmark
  frost train [--epochs N]           train+validate a projection head (save checkpoint)

Model dir: {}"#, model_dir().display());
}

fn require_model() -> Result<()> {
    let d = model_dir();
    if !d.join("model.safetensors").exists() {
        bail!("model not found at {} — run bootstrap.sh to download weights", d.display());
    }
    Ok(())
}

fn diagnose() -> Result<()> {
    println!("FROST diagnose");
    let d = model_dir();
    let has_model = d.join("model.safetensors").exists();
    println!("  model dir       : {}", d.display());
    println!("  weights present : {}", yn(has_model));
    if !has_model { println!("  -> run bootstrap.sh to fetch weights"); return Ok(()); }
    let eng = Engine::load(&d)?;
    println!("  model loaded    : yes");
    println!("  fingerprint     : {}", eng.fingerprint());
    println!("  scorer          : {}", eng.scorer());
    // exercise a real decision
    let req = Request {
        state: "play some music".into(),
        candidates: vec![
            Candidate { id: "play".into(), text: "start playing a song".into(), permission: None },
            Candidate { id: "delete".into(), text: "erase all files".into(), permission: None },
        ],
        mode: Mode::Quiet,
        allowed: None,
    };
    match eng.rank(&req) {
        Decision::Ranked { ranking, pick, trace } => {
            println!("  self-test rank  : top={} pick={:?} device={}", ranking[0].id, pick, trace.device);
            println!("  status          : OK");
        }
        Decision::Deferred { reason, .. } => println!("  self-test       : deferred {reason:?}"),
    }
    Ok(())
}

fn models() -> Result<()> {
    require_model()?;
    let d = model_dir();
    let size = std::fs::metadata(d.join("model.safetensors")).map(|m| m.len()).unwrap_or(0);
    println!("backbone   : nomic-embed-text-v1.5 (reference/fallback ranker)");
    println!("revision   : {}", frost_model::REVISION);
    println!("weights    : {} ({:.1} MiB)", d.join("model.safetensors").display(), size as f64 / 1048576.0);
    let eng = Engine::load(&d)?;
    println!("fingerprint: {}", eng.fingerprint());
    Ok(())
}

fn rank(args: &[String]) -> Result<()> {
    require_model()?;
    let mut state = String::new();
    let mut cands = Vec::new();
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--state" => { i += 1; state = args.get(i).cloned().unwrap_or_default(); }
            "--action" => {
                i += 1;
                let a = args.get(i).cloned().unwrap_or_default();
                let (id, text) = a.split_once('=').unwrap_or(("", a.as_str()));
                cands.push(Candidate { id: if id.is_empty() { format!("a{}", cands.len()) } else { id.into() }, text: text.into(), permission: None });
            }
            "--json" => json = true,
            other => bail!("unknown flag {other}"),
        }
        i += 1;
    }
    if state.is_empty() { bail!("--state is required"); }
    if cands.is_empty() { bail!("at least one --action is required"); }
    let eng = Engine::load(&model_dir())?;
    let decision = eng.rank(&Request { state, candidates: cands, mode: Mode::Quiet, allowed: None });
    if json {
        println!("{}", serde_json::to_string_pretty(&decision)?);
    } else {
        match decision {
            Decision::Ranked { ranking, pick, trace } => {
                println!("pick: {}", pick.unwrap_or_else(|| "(abstained — top too close)".into()));
                println!("ranking:");
                for (r, s) in ranking.iter().enumerate() {
                    println!("  {}. {:<16} {:.4}", r + 1, s.id, s.raw_score);
                }
                println!("trace: exit={:?} width={:?} device={} cache_hit={} scorer via notes={:?}",
                    trace.exit, trace.width, trace.device, trace.cache_hit, trace.notes);
                for (stage, ms) in &trace.stage_ms { println!("  {stage}: {ms:.1} ms"); }
            }
            Decision::Deferred { reason, .. } => println!("deferred: {reason:?}"),
        }
    }
    Ok(())
}

fn bench(args: &[String]) -> Result<()> {
    require_model()?;
    let n: usize = args.iter().position(|a| a == "--n").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(20);
    let emb = Embedder::load(&model_dir())?;
    // warm
    let _ = emb.encode("warmup", frost_model::Task::SearchQuery);
    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let t = Instant::now();
        let _ = emb.encode(&format!("benchmark input number {i} about various topics"), frost_model::Task::SearchDocument);
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| ts[((n as f64 - 1.0) * q).round() as usize];
    println!("encode latency over {n} runs (warm):");
    println!("  p50 = {:.1} ms", p(0.50));
    println!("  p95 = {:.1} ms", p(0.95));
    println!("  min = {:.1} ms  max = {:.1} ms", ts[0], ts[n - 1]);
    Ok(())
}

fn train(args: &[String]) -> Result<()> {
    require_model()?;
    let epochs: usize = args.iter().position(|a| a == "--epochs").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(150);
    let emb = Embedder::load(&model_dir())?;
    let heads_dir = std::path::PathBuf::from(std::env::var("HOME")?).join("Library/Application Support/FROST/heads");
    std::fs::create_dir_all(&heads_dir)?;
    let ckpt = heads_dir.join("full_head.json");
    println!("training projection head ({epochs} epochs) on bundled synthetic routing dataset…");
    let rep = frost_train::pipeline::train_and_eval(&emb, &ckpt, 1234, epochs, 0.05, 0.05)?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    println!("checkpoint: {}", ckpt.display());
    println!("note: head is trained+validated; reference full-depth ranker stays enabled by default");
    Ok(())
}

fn yn(b: bool) -> &'static str { if b { "yes" } else { "no" } }

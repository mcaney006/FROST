//! `frost eval`: run a fixture task through the real engine and the real approval loop, auto-
//! approving every proposed patch/command (the human is simulated by "approve"), then judge the
//! result with the fixture's HELD-OUT test and write every attempt, message and command output to
//! a JSON report. Plumbing outcomes (did the loop run, were attempts proposed/applied/executed)
//! are separated from model quality (did the held-out test pass, first attempt or after repair).

use anyhow::{bail, Context, Result};
use frost_chat::{Engine, Event, Status};
use frost_store::{AttemptKind, AttemptStatus};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct EvalFlags { pub fixture: PathBuf, pub out: PathBuf, pub max_rounds: usize, pub budget: Duration }

pub fn flags(args: &[String]) -> Result<EvalFlags> {
    let mut f = EvalFlags { fixture: PathBuf::new(), out: PathBuf::from("verification/coding_eval"), max_rounds: 12, budget: Duration::from_secs(900) };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--fixture" => { i += 1; f.fixture = PathBuf::from(args.get(i).cloned().unwrap_or_default()); }
            "--out" => { i += 1; f.out = PathBuf::from(args.get(i).cloned().unwrap_or_default()); }
            "--max-rounds" => { i += 1; f.max_rounds = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(12); }
            "--budget-s" => { i += 1; f.budget = Duration::from_secs(args.get(i).and_then(|s| s.parse().ok()).unwrap_or(900)); }
            other => bail!("unknown flag {other}"),
        }
        i += 1;
    }
    if f.fixture.as_os_str().is_empty() { bail!("--fixture DIR is required"); }
    Ok(f)
}

fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let name = e.file_name();
        let n = name.to_string_lossy();
        if n == "target" || n == ".zig-cache" || n == "zig-out" || n == ".git" { continue; }
        let (s, d) = (e.path(), dst.join(&name));
        if e.file_type()?.is_dir() { copy_tree(&s, &d)?; } else { std::fs::copy(&s, &d)?; }
    }
    Ok(())
}

fn sha256_file(p: &Path) -> String {
    frost_tools::Workspace::open(p.parent().unwrap_or(Path::new("."))).ok()
        .and_then(|ws| ws.read(&p.file_name().unwrap().to_string_lossy(), 1, usize::MAX / 2, usize::MAX / 2).ok())
        .map(|r| r.sha256_hex).unwrap_or_default()
}

pub fn run(f: EvalFlags) -> Result<()> {
    let name = f.fixture.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or("fixture".into());
    let expected: Value = serde_json::from_str(&std::fs::read_to_string(f.fixture.join("EXPECTED.json")).context("EXPECTED.json")?)?;
    let task = std::fs::read_to_string(f.fixture.join("TASK.md")).context("TASK.md")?;
    let test_argv: Vec<String> = serde_json::from_value(expected["test"].clone()).context("EXPECTED.test")?;
    let build_argv: Vec<String> = serde_json::from_value(expected["build"].clone()).unwrap_or_default();
    let held_out: Vec<String> = serde_json::from_value(expected["held_out"].clone()).unwrap_or_default();

    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let scratch = std::env::temp_dir().join(format!("frost-eval-{name}-{n}"));
    let work = scratch.join("repo");
    let data = scratch.join("data");
    copy_tree(&f.fixture, &work)?;
    std::fs::create_dir_all(&data)?;
    let real_models = PathBuf::from(std::env::var("HOME")?).join("Library/Application Support/FROST/models");
    std::os::unix::fs::symlink(&real_models, data.join("models"))?;
    let held_before: Vec<(String, String)> = held_out.iter().map(|h| (h.clone(), sha256_file(&work.join(h)))).collect();

    eprintln!("[eval] fixture {name}: work dir {}", work.display());
    let engine = Engine::new(&data)?;
    let events: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    { let ev = events.clone(); engine.subscribe(Box::new(move |e: &Event| { if let Ok(v) = serde_json::to_value(e) { ev.lock().unwrap().push(v); } })); }
    if !engine.wait_idle(Duration::from_secs(180)) { bail!("model did not load in time"); }
    if let Status::Error { detail } = engine.status() { bail!("engine unavailable: {detail}"); }
    let identity = engine.identity();

    let conv = engine.create_conversation()?;
    engine.set_repo(&conv.id, Some(&work.display().to_string()))?;
    let prompt = format!(
        "{task}\n\nTask: fix this bug in the attached repository. Start by reading the relevant source file with read_file, \
then change it with edit_file (exact old text -> new text; the user reviews it as a diff), then verify by running the project's \
build and test commands with run_command (build: {build_argv:?}; test: {test_argv:?}). Do not modify the test files. \
When the test passes, summarize what you changed.");
    let t0 = Instant::now();
    engine.send(&conv.id, &prompt)?;

    let mut rounds = 0usize;
    let mut approved = 0usize;
    loop {
        if !engine.wait_idle(f.budget) { engine.cancel(); bail!("generation exceeded the time budget"); }
        if t0.elapsed() > f.budget { eprintln!("[eval] time budget exhausted"); break; }
        let pending: Vec<_> = engine.attempts(&conv.id)?.into_iter().filter(|a| a.status == AttemptStatus::Proposed).collect();
        if pending.is_empty() { break; }
        rounds += 1;
        if rounds > f.max_rounds { eprintln!("[eval] max rounds reached"); break; }
        for a in pending {
            eprintln!("[eval] auto-approving {} ({})", a.kind.as_str(), a.payload["summary"].as_str().or(a.payload["purpose"].as_str()).unwrap_or(""));
            approved += 1;
            engine.decide_attempt(&conv.id, &a.id, true)?;
            if !engine.wait_idle(f.budget) { engine.cancel(); bail!("attempt exceeded the time budget"); }
        }
    }
    let elapsed_ms = t0.elapsed().as_millis();
    let messages = engine.messages(&conv.id)?;
    let attempts = engine.attempts(&conv.id)?;
    engine.shutdown();

    // Held-out judgement: the fixture's own build (if any) then its test command, in the patched
    // tree, through the same bounded runner; the held-out files must be untouched.
    let ws = frost_tools::Workspace::open(&work)?;
    let spec_for = |argv: Vec<String>| frost_tools::RunSpec { argv, cwd: ".".into(), timeout: Duration::from_secs(300), max_output_bytes: 32 * 1024,
        env_allowlist: ["PATH", "HOME", "TMPDIR", "LANG", "CARGO_HOME", "RUSTUP_HOME"].iter().map(|s| s.to_string()).collect() };
    let never = std::sync::atomic::AtomicBool::new(false);
    let build = if build_argv.is_empty() { None } else { Some(frost_tools::run(&ws, &spec_for(build_argv.clone()), &never)?) };
    let judge = match &build { Some(b) if b.exit_code != Some(0) => b.clone(), _ => frost_tools::run(&ws, &spec_for(test_argv.clone()), &never)? };
    let held_unchanged = held_before.iter().all(|(h, before)| &sha256_file(&work.join(h)) == before);
    let pass = judge.exit_code == Some(0) && held_unchanged;

    let patches_applied = attempts.iter().filter(|a| a.kind == AttemptKind::ProposeDiff && a.status == AttemptStatus::Applied).count();
    let runs: Vec<&_> = attempts.iter().filter(|a| a.kind == AttemptKind::RunCommand).collect();
    let first_run_passed = runs.first().map(|a| a.status == AttemptStatus::Passed).unwrap_or(false);
    let first_attempt_pass = pass && patches_applied == 1 && first_run_passed;
    let repaired_pass = pass && !first_attempt_pass;
    let plumbing_ok = patches_applied >= 1 && !runs.is_empty();

    let report = json!({
        "fixture": name, "fixture_dir": f.fixture.display().to_string(), "task": task,
        "generator": identity, "sampler_backend": frost_kernels::backend(),
        "elapsed_ms": elapsed_ms, "approval_rounds": rounds, "auto_approved": approved,
        "plumbing": { "status": if plumbing_ok { "PASS" } else { "FAIL" }, "patches_applied": patches_applied, "commands_run": runs.len(),
                      "tool_calls_total": messages.iter().filter_map(|m| m.meta["tool_calls"].as_array().map(|a| a.len())).sum::<usize>() },
        "quality": { "held_out_pass": pass, "first_attempt_pass": first_attempt_pass, "repaired_pass": repaired_pass,
                     "held_out_files_unchanged": held_unchanged, "judge": { "argv": judge.argv, "exit_code": judge.exit_code, "timed_out": judge.timed_out,
                     "stdout_tail": judge.stdout.chars().rev().take(1500).collect::<String>().chars().rev().collect::<String>(),
                     "stderr_tail": judge.stderr.chars().rev().take(1500).collect::<String>().chars().rev().collect::<String>() } },
        "attempts": attempts, "messages": messages, "events": events.lock().unwrap().len(),
    });
    std::fs::create_dir_all(&f.out)?;
    let out = f.out.join(format!("{name}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("{name}: plumbing {} | held-out {} ({}) | patches applied {} | commands {} | {:.0}s | report {}",
        if plumbing_ok { "PASS" } else { "FAIL" }, if pass { "PASS" } else { "FAIL" },
        if first_attempt_pass { "first attempt" } else if repaired_pass { "after repair" } else { "not solved" },
        patches_applied, runs.len(), elapsed_ms as f64 / 1e3, out.display());
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(())
}

//! frost — command-line front end over the same generator/engine the app uses.
mod eval;
use anyhow::{bail, Result};
use frost_model::Embedder;
use std::path::PathBuf;
use std::time::Instant;

fn data_dir() -> PathBuf { PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Library/Application Support/FROST") }

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
        "diagnose" => diagnose(&args[1..]),
        "models" => models(),
        "gen" => gen(&args[1..]),
        "chat" => chat(&args[1..]),
        "train" => train(&args[1..]),
        "mode" => set_mode(&args[1..]),
        "eval" => eval::run(eval::flags(&args[1..])?),
        "help" | "-h" | "--help" => { print_help(); Ok(()) }
        other => { eprintln!("unknown command: {other}\n"); print_help(); std::process::exit(2); }
    }
}

fn print_help() {
    println!(r#"frost — local coding chatbot (Ministral-3-8B on MLX) with nomic repository memory

USAGE:
  frost diagnose [--json] [--load]  check components; --load also loads the generator and generates one reply
  frost models                      show generator + retrieval model identity and provenance (no load)
  frost gen --prompt TEXT [--max N] [--temp T] [--system TEXT] [--json]
                                    one streamed generation from the local LLM
  frost chat [--max N] [--temp T]   interactive multi-turn chat (/reset, /stats, /quit)
  frost eval --fixture DIR [--out DIR] [--max-rounds N] [--budget-s S]
                                    run a coding task through the real approval loop (auto-approve) and judge it with the held-out test
  frost mode quiet|balanced|performance [--if-unset]
                                    set the app's scheduling mode (persisted; the app reads it on launch)
  frost train [--epochs N]          (optional experiment) train a retrieval projection head; thermal-governed

Data dir: {}"#, data_dir().display());
}

// ---------------------------------------------------------------------------------------------
// diagnose / models

fn check(name: &str, ok: bool, detail: String, out: &mut Vec<serde_json::Value>) {
    let status = if ok { "PASS" } else { "FAIL" };
    println!("  {status:4} {name:28} {detail}");
    out.push(serde_json::json!({"check": name, "status": status, "detail": detail}));
}

fn diagnose(args: &[String]) -> Result<()> {
    let json = args.iter().any(|a| a == "--json");
    let load = args.iter().any(|a| a == "--load");
    let mut out = Vec::new();
    println!("FROST diagnose");
    let gen_dir = frost_gen::Generator::default_dir();
    match frost_gen::Generator::describe(&gen_dir) {
        Ok(id) => check("generator checkpoint", id.revision == frost_gen::REVISION,
            format!("{} @ {} | {} | {:.2} GiB text weights", id.repo, &id.revision[..id.revision.len().min(12)], id.quantization, id.weight_bytes as f64 / (1u64 << 30) as f64), &mut out),
        Err(e) => check("generator checkpoint", false, format!("{e} (run tools/fetch_ministral.sh)"), &mut out),
    }
    let digests_ok = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "tokenizer.json"].iter().all(|f| gen_dir.join(format!("{f}.sha256")).exists());
    check("generator digests", digests_ok, if digests_ok { "sha256 markers present for shards + tokenizer".into() } else { "missing .sha256 markers".into() }, &mut out);
    match frost_gen::Tokenizer::load(&gen_dir.join("tokenizer.json")) {
        Ok(t) => check("tokenizer", t.vocab_size == 131072, format!("vocab {} , control ids 0..1000 verified", t.vocab_size), &mut out),
        Err(e) => check("tokenizer", false, e.to_string(), &mut out),
    }
    let emb_dir = Embedder::default_dir();
    check("retrieval encoder", emb_dir.join("model.safetensors").exists(), format!("nomic-embed-text-v1.5 @ {} at {}", &frost_model::REVISION[..12], emb_dir.display()), &mut out);
    check("mlx metal", frost_mlx::metal_available(), format!("mlx active {:.0} MiB", frost_mlx::memory().active as f64 / 1048576.0), &mut out);
    // Zig index self-test: write, open, search a tiny index
    let zig = (|| -> Result<String> {
        let dir = std::env::temp_dir().join(format!("frost-diag-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("t.frostidx");
        let vecs: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        frost_index::write_index(&path, &[7, 8, 9], &vecs, 3, 1)?;
        let idx = frost_index::Index::open(&path)?;
        let hits = idx.search(&[0.0, 1.0, 0.0], 1)?;
        let _ = std::fs::remove_dir_all(&dir);
        if hits.first().map(|h| h.0) != Some(8) { bail!("wrong hit {hits:?}"); }
        Ok(format!("write/open/search ok; meta {:?}", idx.meta()))
    })();
    match zig { Ok(d) => check("zig index (C ABI)", true, d, &mut out), Err(e) => check("zig index (C ABI)", false, e.to_string(), &mut out) }
    let logits: Vec<f32> = (0..131072).map(|i| ((i * 7919) % 10007) as f32 / 100.0).collect();
    let k = frost_kernels::topk(&logits, 8).map_err(|e| anyhow::anyhow!("{e}"))?;
    let r = frost_kernels::reference::topk(&logits, 8).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mojo_ok = frost_kernels::backend() == "mojo-dylib" && k == r;
    // Mojo is optional unless FROST_REQUIRE_MOJO=1 (acceptance sets it): an absent dylib passes with a
    // visible "unavailable" detail; a loaded dylib that disagrees with the reference always fails.
    let required = std::env::var("FROST_REQUIRE_MOJO").map(|v| v == "1").unwrap_or(false);
    let pass = mojo_ok || (!required && frost_kernels::backend() != "mojo-dylib");
    check("mojo sampling kernels", pass, format!("backend {} {}{}", frost_kernels::backend(), frost_kernels::load_error().map(|e| format!("({e})")).unwrap_or_else(|| if k == r { "parity vs reference ok".into() } else { "PARITY MISMATCH".into() }), if mojo_ok { "" } else if required { " [required by FROST_REQUIRE_MOJO=1]" } else { " [optional; Rust reference sampler will be used]" }), &mut out);
    let avail = frost_platform::available_memory_bytes().unwrap_or(0);
    check("platform", true, format!("thermal {:?}, pressure {:?}, low-power {}, physical {:.1} GiB, available {:.1} GiB",
        frost_platform::thermal_state(), frost_platform::memory_pressure(), frost_platform::low_power_mode(),
        frost_platform::physical_memory_bytes() as f64 / (1u64 << 30) as f64, avail as f64 / (1u64 << 30) as f64), &mut out);
    if load {
        let mut g = load_generator()?;
        let prompt = frost_gen::render(g.tokenizer(), frost_gen::DEFAULT_SYSTEM_PROMPT, None, &[frost_gen::Message::user("Reply with the single word: ready")])?;
        let (ids, st) = stream_turn(&mut g, &prompt, 8, 0.0)?;
        let text = g.tokenizer().decode(&ids).to_lowercase();
        check("generation", text.contains("ready"), format!("{:?} at {:.1} tok/s, sampler {}", text.trim(), st.tokens_per_second, st.sampler), &mut out);
    }
    let all_ok = out.iter().all(|c| c["status"] == "PASS");
    println!("  => {}", if all_ok { "all checks PASS" } else { "some checks FAIL" });
    if json { println!("{}", serde_json::to_string_pretty(&out)?); }
    if !all_ok { std::process::exit(3); }
    Ok(())
}

fn models() -> Result<()> {
    let gen_dir = frost_gen::Generator::default_dir();
    println!("generator (chat):");
    match frost_gen::Generator::describe(&gen_dir) {
        Ok(id) => println!("{}", serde_json::to_string_pretty(&id)?),
        Err(e) => println!("  not available: {e}"),
    }
    println!("retrieval encoder (repository search only, never chat text):");
    println!("  nomic-ai/nomic-embed-text-v1.5 @ {} at {}", frost_model::REVISION, Embedder::default_dir().display());
    println!("sampling kernels: {}{}", frost_kernels::backend(), frost_kernels::load_error().map(|e| format!(" ({e})")).unwrap_or_default());
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// gen / chat

struct GenFlags { max: usize, temp: f32, system: String, prompt: String, json: bool }
fn gen_flags(args: &[String]) -> Result<GenFlags> {
    let mut f = GenFlags { max: 512, temp: 0.15, system: frost_gen::DEFAULT_SYSTEM_PROMPT.into(), prompt: String::new(), json: false };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => { i += 1; f.prompt = args.get(i).cloned().unwrap_or_default(); }
            "--max" => { i += 1; f.max = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(512); }
            "--temp" => { i += 1; f.temp = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.15); }
            "--system" => { i += 1; f.system = args.get(i).cloned().unwrap_or_default(); }
            "--json" => f.json = true,
            other => bail!("unknown flag {other}"),
        }
        i += 1;
    }
    Ok(f)
}

fn load_generator() -> Result<frost_gen::Generator> {
    let dir = frost_gen::Generator::default_dir();
    let t = Instant::now();
    let g = frost_gen::Generator::load(&dir)
        .map_err(|e| anyhow::anyhow!("generator not loadable from {}: {e}\n  -> run tools/fetch_ministral.sh (bootstrap does this)", dir.display()))?;
    let id = &g.identity;
    eprintln!("[frost] generator: {} @ {} | {} | {} layers, {} kv heads | budget {} tokens | backend {} | weights {:.2} GiB | loaded in {:.1}s",
        id.repo, &id.revision[..12], id.quantization, id.layers, id.kv_heads, id.context_budget, id.backend,
        id.weight_bytes as f64 / (1u64 << 30) as f64, t.elapsed().as_secs_f64());
    Ok(g)
}

fn stream_turn(g: &mut frost_gen::Generator, prompt: &[u32], max: usize, temp: f32) -> Result<(Vec<u32>, frost_gen::Stats)> {
    use std::io::Write;
    let p = frost_gen::GenParams { max_new_tokens: max, sample: frost_gen::SampleParams { temperature: temp, ..Default::default() }, ..Default::default() };
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let mut out = std::io::stdout().lock();
    let r = g.generate(prompt, &p, &cancel, &mut |e| match e {
        frost_gen::Event::Token { text, .. } => { let _ = out.write_all(text.as_bytes()); let _ = out.flush(); }
        frost_gen::Event::Prefilled { tokens, ms } => eprintln!("[frost] prefill {tokens} tokens in {ms:.0} ms"),
        frost_gen::Event::PrefillChunk { .. } => {}
    })?;
    let _ = out.write_all(b"\n");
    Ok(r)
}

fn report(st: &frost_gen::Stats) {
    eprintln!("[frost] {} new tokens in {:.0} ms = {:.1} tok/s | finish {:?} | prefix reused {} | mlx active {:.2} GiB peak {:.2} GiB | kv {:.0} MiB | sampler {}",
        st.new_tokens, st.decode_ms, st.tokens_per_second, st.finish, st.reused_prefix_tokens,
        st.mlx_active_bytes as f64 / (1u64 << 30) as f64, st.mlx_peak_bytes as f64 / (1u64 << 30) as f64,
        st.kv_cache_bytes as f64 / (1u64 << 20) as f64, st.sampler);
}

fn gen(args: &[String]) -> Result<()> {
    let f = gen_flags(args)?;
    if f.prompt.is_empty() { bail!("--prompt is required"); }
    let mut g = load_generator()?;
    let msgs = [frost_gen::Message::user(f.prompt.clone())];
    let prompt = frost_gen::render(g.tokenizer(), &f.system, None, &msgs)?;
    let (ids, st) = stream_turn(&mut g, &prompt, f.max, f.temp)?;
    report(&st);
    if f.json {
        let (text, calls) = frost_gen::parse_output(g.tokenizer(), &ids);
        println!("{}", serde_json::json!({ "identity": g.identity, "stats": st, "text": text, "tool_calls": calls.len() }));
    }
    Ok(())
}

fn chat(args: &[String]) -> Result<()> {
    use std::io::{BufRead, Write};
    let f = gen_flags(args)?;
    let mut g = load_generator()?;
    let mut msgs: Vec<frost_gen::Message> = Vec::new();
    let stdin = std::io::stdin();
    eprintln!("[frost] chat ready. /reset clears history, /stats shows memory, /quit exits.");
    loop {
        eprint!("you> "); let _ = std::io::stderr().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 { break; }
        let line = line.trim();
        match line {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => { msgs.clear(); g.reset(); eprintln!("[frost] history and KV cache cleared"); continue; }
            "/stats" => { let m = frost_mlx::memory(); eprintln!("[frost] mlx active {:.2} GiB, peak {:.2} GiB, cache {:.0} MiB; kv {:.0} MiB; {} messages",
                m.active as f64 / (1u64 << 30) as f64, m.peak as f64 / (1u64 << 30) as f64, m.cache as f64 / (1u64 << 20) as f64,
                g.kv_cache_bytes() as f64 / (1u64 << 20) as f64, msgs.len()); continue; }
            _ => {}
        }
        msgs.push(frost_gen::Message::user(line.to_string()));
        let prompt = match frost_gen::render(g.tokenizer(), &f.system, None, &msgs) {
            Ok(p) => p,
            Err(e) => { eprintln!("[frost] {e}"); msgs.pop(); continue; }
        };
        if prompt.len() + f.max > frost_gen::CONTEXT_BUDGET {
            eprintln!("[frost] context budget: prompt {} + {} reserved > {}. Use /reset or shorten the message.", prompt.len(), f.max, frost_gen::CONTEXT_BUDGET);
            msgs.pop();
            continue;
        }
        eprint!("frost> ");
        let (ids, st) = stream_turn(&mut g, &prompt, f.max, f.temp)?;
        report(&st);
        let (text, _) = frost_gen::parse_output(g.tokenizer(), &ids);
        msgs.push(frost_gen::Message::assistant(text));
    }
    Ok(())
}

fn set_mode(args: &[String]) -> Result<()> {
    let m = args.first().map(String::as_str).unwrap_or("");
    if !matches!(m, "quiet" | "balanced" | "performance") { bail!("usage: frost mode quiet|balanced|performance [--if-unset]"); }
    std::fs::create_dir_all(data_dir())?;
    let st = frost_store::Store::open(&data_dir().join("frost.sqlite"))?;
    // --if-unset: the installer's default; never overrides a mode the user already chose.
    if args.iter().any(|a| a == "--if-unset") {
        if let Some(existing) = st.get_setting("mode")? { println!("mode = {existing} (kept)"); return Ok(()); }
    }
    st.set_setting("mode", m)?;
    println!("mode = {m}");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// train (optional experiment; not part of the chat product)

fn train(args: &[String]) -> Result<()> {
    let dir = Embedder::default_dir();
    if !dir.join("model.safetensors").exists() { bail!("retrieval encoder not found at {}", dir.display()); }
    // Training is background work: it must not bypass the thermal governor.
    let t = frost_platform::thermal_state();
    if !matches!(t, frost_core::ThermalState::Nominal) { bail!("thermal state is {t:?}; training only runs when the system is Nominal"); }
    let epochs: usize = args.iter().position(|a| a == "--epochs").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(150);
    let emb = Embedder::load(&dir)?;
    let heads_dir = data_dir().join("heads");
    std::fs::create_dir_all(&heads_dir)?;
    let ckpt = heads_dir.join("full_head.json");
    println!("training projection head ({epochs} epochs) on the bundled synthetic routing dataset (experiment; not used by chat)…");
    let rep = frost_train::pipeline::train_and_eval(&emb, &ckpt, 1234, epochs, 0.05, 0.05)?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    println!("checkpoint: {}", ckpt.display());
    Ok(())
}

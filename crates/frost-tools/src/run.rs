//! Bounded process runner. See the crate docs: this bounds lifetime, output,
//! environment leakage and stdin — it is NOT a sandbox.

use crate::{ToolError, Workspace};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// A sensible allowlist for toolchain commands.
pub const DEFAULT_ENV_ALLOWLIST: &[&str] = &["PATH", "HOME", "CARGO_HOME", "RUSTUP_HOME", "TMPDIR", "LANG"];
/// PATH used when `PATH` is not allowlisted (or unset in the parent);
/// `$HOME/.cargo/bin` and `$HOME/.local/bin` are appended when HOME is known.
pub const SAFE_PATH_DIRS: &[&str] = &["/usr/bin", "/bin", "/usr/sbin", "/sbin", "/opt/homebrew/bin"];

const POLL: Duration = Duration::from_millis(20);
const TERM_GRACE: Duration = Duration::from_secs(2);
/// After the process group is dead, how long to wait for pipe readers to drain.
/// Only a descendant that escaped the group (setsid) can keep a pipe open longer.
const READER_GRACE: Duration = Duration::from_secs(2);
const SIGKILL: i32 = 9;
const SIGTERM: i32 = 15;

extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

fn kill_group(pgid: i32, sig: i32) {
    // 0 would address our own group and -1 every process we may signal: never.
    if pgid > 1 {
        // SAFETY: kill(2) takes plain integers and has no memory-safety preconditions.
        unsafe {
            kill(-pgid, sig);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSpec {
    /// `argv[0]`: absolute path, a bare name looked up on the child's PATH, or
    /// `./name` resolved inside the workspace relative to `cwd`. Never a shell.
    pub argv: Vec<String>,
    /// Workspace-relative working directory (`"."` = root).
    pub cwd: String,
    pub timeout: Duration,
    /// Per stream: the first half and the last half are kept.
    pub max_output_bytes: usize,
    /// Parent env vars copied into the otherwise empty child env.
    pub env_allowlist: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunResult {
    /// argv as executed (argv[0] resolved to an absolute path).
    pub argv: Vec<String>,
    /// Absolute working directory.
    pub cwd: String,
    /// Exactly the env keys the child received, sorted.
    pub env_keys: Vec<String>,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub timed_out: bool,
    pub cancelled: bool,
    pub duration_ms: u64,
}

/// Head + tail capture with a hard byte cap. The pipe is always drained, so a
/// chatty child never blocks on a full pipe.
struct Capture {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    total: usize,
    half: usize,
}

impl Capture {
    fn new(max: usize) -> Capture {
        Capture { head: Vec::new(), tail: VecDeque::new(), total: 0, half: max / 2 }
    }

    fn push(&mut self, chunk: &[u8]) {
        self.total += chunk.len();
        let take = (self.half - self.head.len()).min(chunk.len());
        self.head.extend_from_slice(&chunk[..take]);
        self.tail.extend(&chunk[take..]);
        if self.tail.len() > self.half {
            let excess = self.tail.len() - self.half;
            self.tail.drain(..excess);
        }
    }

    fn finish(&self) -> (String, bool) {
        let dropped = self.total - self.head.len() - self.tail.len();
        let mut out = self.head.clone();
        if dropped > 0 {
            out.extend_from_slice(format!("\n…[truncated {dropped} bytes]…\n").as_bytes());
        }
        out.extend(self.tail.iter());
        (String::from_utf8_lossy(&out).into_owned(), dropped > 0)
    }
}

fn spawn_reader(mut r: impl Read + Send + 'static, cap: Arc<Mutex<Capture>>) -> Receiver<()> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut buf = [0u8; 16 * 1024];
        loop {
            match r.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => cap.lock().unwrap_or_else(|p| p.into_inner()).push(&buf[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        let _ = tx.send(());
    });
    rx
}

fn child_env(allowlist: &[String]) -> Result<BTreeMap<String, String>, ToolError> {
    let mut env = BTreeMap::new();
    for key in allowlist {
        if key.is_empty() || key.contains('=') || key.contains('\0') {
            return Err(ToolError::BadCommand { msg: format!("invalid env allowlist key {key:?}") });
        }
        if let Ok(v) = std::env::var(key) {
            env.insert(key.clone(), v);
        }
    }
    if !env.contains_key("PATH") {
        let mut dirs: Vec<String> = SAFE_PATH_DIRS.iter().map(|s| s.to_string()).collect();
        if let Some(home) = std::env::var("HOME").ok().filter(|h| h.starts_with('/')) {
            dirs.push(format!("{home}/.cargo/bin"));
            dirs.push(format!("{home}/.local/bin"));
        }
        env.insert("PATH".into(), dirs.join(":"));
    }
    Ok(env)
}

fn resolve_program(ws: &Workspace, cwd_rel: &str, argv0: &str, path_var: &str) -> Result<PathBuf, ToolError> {
    let bad = |msg: String| ToolError::BadCommand { msg };
    if argv0.is_empty() {
        return Err(bad("empty argv[0]".into()));
    }
    if argv0.starts_with('/') {
        return Ok(PathBuf::from(argv0));
    }
    if let Some(rest) = argv0.strip_prefix("./") {
        // Confined by the same resolver as every other path (no escapes, no excluded names).
        let p = ws.resolve(&format!("{cwd_rel}/{rest}"))?;
        return if p.is_file() { Ok(p) } else { Err(bad(format!("{argv0} is not a file in {cwd_rel}"))) };
    }
    if argv0.contains('/') {
        return Err(bad(format!("argv[0] {argv0:?} must be absolute, ./-relative to cwd, or a bare name")));
    }
    // Relative PATH entries (e.g. ".") would make lookup cwd-dependent: skipped.
    path_var
        .split(':')
        .filter(|d| d.starts_with('/'))
        .map(|d| PathBuf::from(d).join(argv0))
        .find(|p| p.metadata().is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0))
        .ok_or_else(|| bad(format!("{argv0:?} not found on PATH {path_var}")))
}

/// SIGTERM the group, give it [`TERM_GRACE`], then SIGKILL the group and reap.
fn terminate(child: &mut Child, pgid: i32) -> std::io::Result<ExitStatus> {
    kill_group(pgid, SIGTERM);
    let grace_end = Instant::now() + TERM_GRACE;
    while Instant::now() < grace_end {
        if let Some(s) = child.try_wait()? {
            return Ok(s);
        }
        thread::sleep(POLL);
    }
    kill_group(pgid, SIGKILL);
    let _ = child.kill();
    child.wait()
}

/// Runs `spec` inside `ws` with a scrubbed environment, no stdin, no shell, in
/// its own process group, with capped output and a hard deadline. Setting
/// `cancel` stops it the same way a timeout does. The whole process group is
/// killed when the command ends, so background children do not outlive it.
pub fn run(ws: &Workspace, spec: &RunSpec, cancel: &AtomicBool) -> Result<RunResult, ToolError> {
    if spec.argv.iter().any(|a| a.contains('\0')) {
        return Err(ToolError::BadCommand { msg: "argv contains NUL".into() });
    }
    let argv0 = spec.argv.first().ok_or_else(|| ToolError::BadCommand { msg: "empty argv".into() })?;
    let cwd = ws.resolve(&spec.cwd)?;
    if !cwd.is_dir() {
        return Err(ToolError::BadCommand { msg: format!("cwd {:?} is not a directory", spec.cwd) });
    }
    let env = child_env(&spec.env_allowlist)?;
    let program = resolve_program(ws, &spec.cwd, argv0, &env["PATH"])?;
    let mut argv = spec.argv.clone();
    argv[0] = program.display().to_string();

    let mut result = RunResult {
        argv,
        cwd: cwd.display().to_string(),
        env_keys: env.keys().cloned().collect(),
        exit_code: None,
        signal: None,
        stdout: String::new(),
        stderr: String::new(),
        stdout_truncated: false,
        stderr_truncated: false,
        timed_out: false,
        cancelled: false,
        duration_ms: 0,
    };
    if cancel.load(Ordering::Relaxed) {
        result.cancelled = true; // cancelled before start: nothing spawned
        return Ok(result);
    }

    let start = Instant::now();
    let mut child = Command::new(&program)
        .args(&spec.argv[1..])
        .current_dir(&cwd)
        .env_clear()
        .envs(&env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|e| ToolError::Spawn { program: result.argv[0].clone(), msg: e.to_string() })?;
    let pgid = i32::try_from(child.id()).unwrap_or(0);
    let out_cap = Arc::new(Mutex::new(Capture::new(spec.max_output_bytes)));
    let err_cap = Arc::new(Mutex::new(Capture::new(spec.max_output_bytes)));
    let readers = [
        spawn_reader(child.stdout.take().expect("stdout piped"), out_cap.clone()),
        spawn_reader(child.stderr.take().expect("stderr piped"), err_cap.clone()),
    ];

    let deadline = start + spec.timeout;
    let waited = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Ok(s),
            Ok(None) => {}
            Err(e) => break Err(e),
        }
        if cancel.load(Ordering::Relaxed) {
            result.cancelled = true;
            break terminate(&mut child, pgid);
        }
        if Instant::now() >= deadline {
            result.timed_out = true;
            break terminate(&mut child, pgid);
        }
        thread::sleep(POLL);
    };
    // The group must not outlive the command: stragglers (background jobs,
    // children that ignored SIGTERM) are killed here. ponytail: signalling a
    // just-reaped leader's pgid is safe while any member lives (the id stays
    // reserved); if the group is already empty it is ESRCH unless the pid was
    // recycled within microseconds — acceptable; waitid(WNOWAIT) would close it.
    kill_group(pgid, SIGKILL);
    let status = match waited {
        Ok(s) => s,
        Err(e) => {
            let _ = child.kill();
            child.wait().map_err(|_| ToolError::Spawn { program: result.argv[0].clone(), msg: format!("wait failed: {e}") })?
        }
    };
    let reader_deadline = Instant::now() + READER_GRACE;
    for rx in &readers {
        let _ = rx.recv_timeout(reader_deadline.saturating_duration_since(Instant::now()));
    }

    result.duration_ms = start.elapsed().as_millis() as u64;
    result.exit_code = status.code();
    result.signal = status.signal();
    (result.stdout, result.stdout_truncated) = out_cap.lock().unwrap_or_else(|p| p.into_inner()).finish();
    (result.stderr, result.stderr_truncated) = err_cap.lock().unwrap_or_else(|p| p.into_inner()).finish();
    Ok(result)
}

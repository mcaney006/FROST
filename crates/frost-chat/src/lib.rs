//! frost-chat: the one model-owning engine behind the app and the CLI.
//!
//! Owns conversation state (SQLite via `frost-store`), the generator session (one worker
//! thread, one active generation), the context/admission/duty policy, cancellation, the
//! thermal and memory governors, the approval-gated coding-agent loop (`agent`) and error
//! reporting. UI layers subscribe to `Event`s; token deltas are coalesced so the transcript is
//! not re-laid-out per token.

pub mod agent;
pub mod ffi;

use frost_core::{Mode, ThermalState};
use frost_gen::{ctl, Event as GenEvent, FinishReason, GenParams, Generator, Identity, SampleParams, ToolCall, CONTEXT_BUDGET, DEFAULT_SYSTEM_PROMPT};
use frost_store::{Attempt, AttemptStatus, Conversation, Message, Role, Store};
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Output tokens reserved out of the total budget; history gets the rest.
pub const RESERVED_OUTPUT_TOKENS: usize = 1024;
/// Minimum interval between coalesced delta events.
const DELTA_FLUSH: Duration = Duration::from_millis(40);
/// How often the governor re-reads the OS thermal / memory-pressure state during a generation.
const THERMAL_POLL: Duration = Duration::from_millis(500);
/// Idle time after which the worker releases the model weights (reloaded on the next request).
pub const IDLE_UNLOAD: Duration = Duration::from_secs(10 * 60);

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("store: {0}")]
    Store(#[from] frost_store::StoreError),
    #[error("the model is still loading")]
    Loading,
    #[error("the model is unavailable: {0}")]
    Unavailable(String),
    #[error("a generation is already running")]
    Busy,
    #[error("empty message")]
    Empty,
    #[error("nothing to regenerate")]
    NothingToRegenerate,
    #[error("unknown mode {0:?} (quiet|balanced|performance)")]
    BadMode(String),
    #[error("engine is shut down")]
    Shutdown,
    #[error("another FROST engine already owns {0} (single-instance lock)")]
    AlreadyRunning(String),
    #[error("attempt {0} is not awaiting a decision")]
    NotProposed(String),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Status { Loading { detail: String }, Ready, Error { detail: String } }

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Status { status: Status },
    ConversationsChanged,
    MessagesChanged { conv_id: String },
    MessageStarted { conv_id: String, message_id: String },
    Delta { conv_id: String, message_id: String, text: String },
    MessageDone { conv_id: String, message_id: String, content: String, meta: Value },
    Note { conv_id: String, text: String },
    Error { conv_id: Option<String>, detail: String },
    /// A patch or command is waiting for the user's decision (`frost_attempt_decide`).
    AttemptProposed { conv_id: String, attempt: Value },
    AttemptUpdated { conv_id: String, attempt: Value },
}

enum Cmd { Generate { conv_id: String }, Execute { conv_id: String, attempt_id: String }, Shutdown }

type Sink = Box<dyn Fn(&Event) + Send + Sync>;

struct Inner {
    /// Held for the engine's lifetime: only one engine may own a data dir (and its model copy).
    _lock: std::fs::File,
    store: Arc<Mutex<Store>>,
    tx: Mutex<Option<SyncSender<Cmd>>>,
    cancel: AtomicBool,
    /// Commands queued or running; the engine is busy while this is non-zero (a generation that
    /// re-enqueues itself for a tool round stays busy across the hand-off).
    pending: AtomicUsize,
    mode: Mutex<Mode>,
    status: Mutex<Status>,
    sinks: Mutex<Vec<Sink>>,
    identity: Mutex<Option<Identity>>,
    kv_bytes: AtomicUsize,
    /// Set by the governor after a critical-pressure stop; the worker drops the model afterwards.
    unload_requested: AtomicBool,
    model_loaded: AtomicBool,
    model_dir: PathBuf,
    /// Repository memory (nomic encoder + Zig index), created lazily on the worker when a
    /// conversation with an attached repository first needs it.
    indexer: Mutex<Option<frost_repo::Indexer>>,
    retrieval_error: Mutex<Option<String>>,
    index_dir: PathBuf,
    worker: Mutex<Option<JoinHandle<()>>>,
    #[cfg(any(test, feature = "thermal-sim"))]
    thermal_sim: Mutex<Option<ThermalState>>,
}

#[derive(Clone)]
pub struct Engine(Arc<Inner>);

fn mode_name(m: Mode) -> &'static str { match m { Mode::Quiet => "quiet", Mode::Balanced => "balanced", Mode::Performance => "performance" } }
fn parse_mode(s: &str) -> Option<Mode> { match s { "quiet" => Some(Mode::Quiet), "balanced" => Some(Mode::Balanced), "performance" => Some(Mode::Performance), _ => None } }

impl Engine {
    /// `data_dir` is the FROST Application Support directory (holds `frost.sqlite` and `models/`).
    /// Returns immediately; the generator loads on the worker thread and `Status` reports progress.
    pub fn new(data_dir: &Path) -> Result<Engine, EngineError> {
        std::fs::create_dir_all(data_dir).ok();
        let lock = single_instance_lock(&data_dir.join("frost.lock"))?;
        let store = Store::open(&data_dir.join("frost.sqlite"))?;
        let mode = store.get_setting("mode").ok().flatten().and_then(|s| parse_mode(&s)).unwrap_or_default();
        let (tx, rx) = sync_channel::<Cmd>(1);
        let inner = Arc::new(Inner {
            _lock: lock,
            store: Arc::new(Mutex::new(store)),
            tx: Mutex::new(Some(tx)),
            cancel: AtomicBool::new(false),
            pending: AtomicUsize::new(0),
            mode: Mutex::new(mode),
            status: Mutex::new(Status::Loading { detail: "starting".into() }),
            sinks: Mutex::new(Vec::new()),
            identity: Mutex::new(None),
            kv_bytes: AtomicUsize::new(0),
            unload_requested: AtomicBool::new(false),
            model_loaded: AtomicBool::new(false),
            model_dir: data_dir.join("models").join("Ministral-3-8B-Instruct-2512-4bit"),
            indexer: Mutex::new(None),
            retrieval_error: Mutex::new(None),
            index_dir: data_dir.join("index"),
            worker: Mutex::new(None),
            #[cfg(any(test, feature = "thermal-sim"))]
            thermal_sim: Mutex::new(None),
        });
        let w = { let inner = inner.clone(); std::thread::Builder::new().name("frost-generator".into()).spawn(move || worker(inner, rx)).expect("spawn worker") };
        *inner.worker.lock().unwrap() = Some(w);
        Ok(Engine(inner))
    }

    pub fn subscribe(&self, sink: Sink) { self.0.sinks.lock().unwrap().push(sink); }
    /// Drop every subscriber (the C side passes a NULL callback to unregister).
    pub fn clear_subscribers(&self) { self.0.sinks.lock().unwrap().clear(); }
    pub fn status(&self) -> Status { self.0.status.lock().unwrap().clone() }
    pub fn mode(&self) -> Mode { *self.0.mode.lock().unwrap() }
    pub fn is_generating(&self) -> bool { self.0.pending.load(Ordering::Relaxed) > 0 }
    pub fn identity(&self) -> Option<Identity> { self.0.identity.lock().unwrap().clone() }

    /// Snapshot for status bars: state, mode, thermal state, memory pressure, MLX memory, KV cache bytes.
    pub fn status_json(&self) -> Value {
        let mem = frost_mlx::memory();
        let mut v = serde_json::to_value(self.status()).unwrap_or(Value::Null);
        v["generating"] = json!(self.is_generating());
        v["mode"] = json!(mode_name(self.mode()));
        v["thermal"] = json!(format!("{:?}", self.0.thermal()));
        v["memory_pressure"] = json!(format!("{:?}", frost_platform::memory_pressure()));
        v["available_memory_bytes"] = json!(frost_platform::available_memory_bytes());
        v["mlx_active_bytes"] = json!(mem.active);
        v["mlx_peak_bytes"] = json!(mem.peak);
        v["kv_cache_bytes"] = json!(self.0.kv_bytes.load(Ordering::Relaxed));
        v["context_budget"] = json!(CONTEXT_BUDGET);
        v["reserved_output_tokens"] = json!(RESERVED_OUTPUT_TOKENS);
        v["model_loaded"] = json!(self.0.model_loaded.load(Ordering::Relaxed));
        v["idle_unload_seconds"] = json!(IDLE_UNLOAD.as_secs());
        v
    }

    /// Generator identity and the (separate) retrieval model, never conflated.
    pub fn model_info(&self) -> Value {
        json!({
            "generator": self.identity(),
            "retrieval": { "model": "nomic-ai/nomic-embed-text-v1.5", "revision": frost_model_revision(), "role": "repository search only; never produces chat text" },
            "sampler_backend": frost_kernels::backend(),
        })
    }

    pub fn set_mode(&self, name: &str) -> Result<(), EngineError> {
        let m = parse_mode(name).ok_or_else(|| EngineError::BadMode(name.into()))?;
        *self.0.mode.lock().unwrap() = m;
        self.0.store.lock().unwrap().set_setting("mode", name)?;
        Ok(())
    }

    pub fn conversations(&self) -> Result<Vec<Conversation>, EngineError> { Ok(self.0.store.lock().unwrap().list_conversations()?) }
    pub fn messages(&self, conv_id: &str) -> Result<Vec<Message>, EngineError> { Ok(self.0.store.lock().unwrap().list_messages(conv_id)?) }
    pub fn attempts(&self, conv_id: &str) -> Result<Vec<Attempt>, EngineError> { Ok(self.0.store.lock().unwrap().list_attempts(conv_id)?) }

    pub fn create_conversation(&self) -> Result<Conversation, EngineError> {
        let c = self.0.store.lock().unwrap().create_conversation("New chat", mode_name(self.mode()), DEFAULT_SYSTEM_PROMPT)?;
        self.0.emit(&Event::ConversationsChanged);
        Ok(c)
    }
    pub fn delete_conversation(&self, id: &str) -> Result<(), EngineError> {
        self.0.store.lock().unwrap().delete_conversation(id)?;
        self.0.emit(&Event::ConversationsChanged);
        Ok(())
    }
    pub fn rename_conversation(&self, id: &str, title: &str) -> Result<(), EngineError> {
        self.0.store.lock().unwrap().rename_conversation(id, title)?;
        self.0.emit(&Event::ConversationsChanged);
        Ok(())
    }
    /// Attach (or detach with `None`) the one repository this conversation may read and patch.
    pub fn set_repo(&self, id: &str, path: Option<&str>) -> Result<(), EngineError> {
        let canon = match path { Some(p) => Some(std::fs::canonicalize(p).map_err(|e| EngineError::Unavailable(format!("repository path: {e}")))?.display().to_string()), None => None };
        self.0.store.lock().unwrap().set_repo_path(id, canon.as_deref())?;
        self.0.emit(&Event::ConversationsChanged);
        Ok(())
    }

    /// Append the user's message and queue a generation. Refuses while loading or busy.
    pub fn send(&self, conv_id: &str, text: &str) -> Result<(), EngineError> {
        let text = text.trim();
        if text.is_empty() { return Err(EngineError::Empty); }
        self.ensure_ready()?;
        {
            let mut st = self.0.store.lock().unwrap();
            let conv = st.get_conversation(conv_id)?;
            let first = st.list_messages(conv_id)?.iter().all(|m| m.role != Role::User);
            st.append_message(conv_id, Role::User, text, json!({}))?;
            if first && conv.title == "New chat" {
                let title: String = text.chars().take(48).collect();
                st.rename_conversation(conv_id, title.trim())?;
            }
        }
        self.0.emit(&Event::MessagesChanged { conv_id: conv_id.into() });
        self.0.emit(&Event::ConversationsChanged);
        self.enqueue(Cmd::Generate { conv_id: conv_id.into() })
    }

    /// Drop the last assistant reply (and any tool turns after the last user message) and generate again.
    pub fn regenerate(&self, conv_id: &str) -> Result<(), EngineError> {
        self.ensure_ready()?;
        {
            let st = self.0.store.lock().unwrap();
            let msgs = st.list_messages(conv_id)?;
            let last_user = msgs.iter().rev().find(|m| m.role == Role::User).ok_or(EngineError::NothingToRegenerate)?;
            st.truncate_after(conv_id, last_user.seq)?;
        }
        self.0.emit(&Event::MessagesChanged { conv_id: conv_id.into() });
        self.enqueue(Cmd::Generate { conv_id: conv_id.into() })
    }

    /// Forget everything before this point for the model (the transcript stays visible).
    pub fn clear_context(&self, conv_id: &str) -> Result<(), EngineError> {
        self.0.store.lock().unwrap().append_message(conv_id, Role::System, "context cleared", json!({"marker": "context_cleared"}))?;
        self.0.emit(&Event::MessagesChanged { conv_id: conv_id.into() });
        Ok(())
    }

    /// The user's decision on a proposed patch/command. Approval executes on the worker thread,
    /// records the outcome, feeds it back to the model as a tool result and continues the loop.
    pub fn decide_attempt(&self, conv_id: &str, attempt_id: &str, approve: bool) -> Result<(), EngineError> {
        self.ensure_ready()?;
        let attempt = {
            let st = self.0.store.lock().unwrap();
            let a = st.list_attempts(conv_id)?.into_iter().find(|a| a.id == attempt_id).ok_or_else(|| EngineError::NotProposed(attempt_id.into()))?;
            if a.status != AttemptStatus::Proposed { return Err(EngineError::NotProposed(attempt_id.into())); }
            st.set_attempt_status(attempt_id, if approve { AttemptStatus::Approved } else { AttemptStatus::Denied })?;
            a
        };
        if approve {
            self.0.emit(&Event::AttemptUpdated { conv_id: conv_id.into(), attempt: attempt_json(&attempt, AttemptStatus::Approved) });
            return self.enqueue(Cmd::Execute { conv_id: conv_id.into(), attempt_id: attempt_id.into() });
        }
        let result = json!({"denied": true, "kind": attempt.kind.as_str(), "note": "the user declined this action; do not retry it unchanged"}).to_string();
        self.0.store.lock().unwrap().append_message(conv_id, Role::Tool, &result, json!({"attempt_id": attempt_id, "tool": attempt.kind.as_str()}))?;
        self.0.emit(&Event::AttemptUpdated { conv_id: conv_id.into(), attempt: attempt_json(&attempt, AttemptStatus::Denied) });
        self.0.emit(&Event::MessagesChanged { conv_id: conv_id.into() });
        self.enqueue(Cmd::Generate { conv_id: conv_id.into() })
    }

    pub fn cancel(&self) { self.0.cancel.store(true, Ordering::Relaxed); }

    /// Stop the worker (cancels any generation first) and wait for it. Idempotent.
    pub fn shutdown(&self) {
        self.cancel();
        if let Some(tx) = self.0.tx.lock().unwrap().take() { let _ = tx.send(Cmd::Shutdown); }
        if let Some(h) = self.0.worker.lock().unwrap().take() { let _ = h.join(); }
    }

    /// Block until no generation is running (tests and the CLI).
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let t = Instant::now();
        while self.is_generating() || matches!(self.status(), Status::Loading { .. }) {
            if t.elapsed() > timeout { return false; }
            std::thread::sleep(Duration::from_millis(10));
        }
        true
    }

    #[cfg(any(test, feature = "thermal-sim"))]
    pub fn simulate_thermal(&self, t: Option<ThermalState>) { *self.0.thermal_sim.lock().unwrap() = t; }

    fn ensure_ready(&self) -> Result<(), EngineError> {
        match self.status() {
            Status::Ready => {}
            Status::Loading { .. } => return Err(EngineError::Loading),
            Status::Error { detail } => return Err(EngineError::Unavailable(detail)),
        }
        if self.is_generating() { return Err(EngineError::Busy); }
        Ok(())
    }
    fn enqueue(&self, cmd: Cmd) -> Result<(), EngineError> { self.0.enqueue(cmd) }
}

impl Inner {
    fn emit(&self, e: &Event) { for s in self.sinks.lock().unwrap().iter() { s(e); } }
    fn set_status(&self, s: Status) { *self.status.lock().unwrap() = s.clone(); self.emit(&Event::Status { status: s }); }
    fn thermal(&self) -> ThermalState { self.sim_thermal().unwrap_or_else(frost_platform::thermal_state) }
    #[cfg(any(test, feature = "thermal-sim"))]
    fn sim_thermal(&self) -> Option<ThermalState> { *self.thermal_sim.lock().unwrap() }
    #[cfg(not(any(test, feature = "thermal-sim")))]
    fn sim_thermal(&self) -> Option<ThermalState> { None }

    fn enqueue(&self, cmd: Cmd) -> Result<(), EngineError> {
        let guard = self.tx.lock().unwrap();
        let tx = guard.as_ref().ok_or(EngineError::Shutdown)?;
        self.cancel.store(false, Ordering::Relaxed);
        // Busy from the moment a request is queued, not from when the worker picks it up.
        self.pending.fetch_add(1, Ordering::SeqCst);
        match tx.try_send(cmd) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => { self.pending.fetch_sub(1, Ordering::SeqCst); Err(EngineError::Busy) }
            Err(TrySendError::Disconnected(_)) => { self.pending.fetch_sub(1, Ordering::SeqCst); Err(EngineError::Shutdown) }
        }
    }
    fn done(&self) { let _ = self.pending.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |p| Some(p.saturating_sub(1))); }
}

/// Make sure the repository indexer exists (loads the nomic encoder on first use). Worker thread only.
fn ensure_indexer(inner: &Inner, conv_id: &str) -> bool {
    if inner.indexer.lock().unwrap().is_some() { return true; }
    if inner.retrieval_error.lock().unwrap().is_some() { return false; }
    let dir = frost_model::Embedder::default_dir();
    match std::panic::catch_unwind(|| frost_model::Embedder::load(&dir)) {
        Ok(Ok(emb)) => {
            let emb: Arc<Mutex<dyn frost_repo::Embed>> = Arc::new(Mutex::new(emb));
            std::fs::create_dir_all(&inner.index_dir).ok();
            *inner.indexer.lock().unwrap() = Some(frost_repo::Indexer::new(inner.store.clone(), emb, inner.index_dir.clone()));
            true
        }
        Ok(Err(e)) => { let m = format!("repository search unavailable: retrieval encoder not loadable from {}: {e}", dir.display()); *inner.retrieval_error.lock().unwrap() = Some(m.clone()); inner.emit(&Event::Note { conv_id: conv_id.into(), text: m }); false }
        Err(_) => { let m = "repository search unavailable: retrieval encoder crashed while loading".to_string(); *inner.retrieval_error.lock().unwrap() = Some(m.clone()); inner.emit(&Event::Note { conv_id: conv_id.into(), text: m }); false }
    }
}

/// Index/refresh the attached repository (governor-gated) and retrieve context for `query`.
/// Returns the framed context block and the hits it was built from.
fn retrieve(inner: &Inner, conv_id: &str, root: &Path, query: &str, mode: Mode) -> Option<(frost_repo::ContextBlock, Vec<frost_repo::Hit>)> {
    if !ensure_indexer(inner, conv_id) { return None; }
    let guard = inner.indexer.lock().unwrap();
    let ix = guard.as_ref()?;
    // Indexing is background work: it does not bypass the thermal/memory governor. Quiet pauses at Fair.
    let thermal = inner.thermal();
    let hot = match mode { Mode::Quiet => !matches!(thermal, ThermalState::Nominal), _ => matches!(thermal, ThermalState::Serious | ThermalState::Critical) };
    let pressured = frost_platform::memory_pressure() != frost_platform::MemoryPressure::Normal;
    let changed = ix.changed_files(root).map(|v| v.len()).unwrap_or(usize::MAX);
    if changed > 0 {
        if hot || pressured {
            inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("repository indexing paused ({changed} changed files): thermal {thermal:?}, memory pressure {:?}; answering from the existing index", frost_platform::memory_pressure()) });
        } else {
            let t = Instant::now();
            let mut last = Instant::now();
            let r = if changed > 50 {
                ix.index_repo(root, &mut |p: frost_repo::Progress| {
                    if last.elapsed() > Duration::from_millis(1500) { last = Instant::now(); inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("indexing repository: {}/{} files, {} chunks", p.files_done, p.files_total, p.chunks_done) }); }
                }).map(Some)
            } else { ix.refresh(root) };
            match r {
                Ok(Some(rep)) => inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("repository indexed: {} files, {} chunks ({} newly embedded, {} from cache, {} skipped) in {:.1}s", rep.files_indexed, rep.chunks, rep.embedded_new, rep.embedded_cached, rep.files_skipped.len(), t.elapsed().as_secs_f64()) }),
                Ok(None) => {}
                Err(e) => inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("repository indexing failed: {e}") }),
            }
        }
    }
    match ix.search(root, query, 6) {
        Ok(hits) if !hits.is_empty() => Some((frost_repo::context_block(&hits, 6000), hits)),
        Ok(_) => None,
        Err(e) => { inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("repository search failed: {e}") }); None }
    }
}

fn attempt_json(a: &Attempt, status: AttemptStatus) -> Value {
    let mut v = serde_json::to_value(a).unwrap_or(Value::Null);
    v["status"] = json!(status.as_str());
    v
}

fn frost_model_revision() -> &'static str { "e9b6763023c676ca8431644204f50c2b100d9aab" }

extern "C" { fn flock(fd: i32, op: i32) -> i32; }
const LOCK_EX: i32 = 2;
const LOCK_NB: i32 = 4;

/// Advisory exclusive lock on `path`; released when the returned file is dropped (process exit included).
fn single_instance_lock(path: &Path) -> Result<std::fs::File, EngineError> {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(path).map_err(|e| EngineError::AlreadyRunning(format!("{}: {e}", path.display())))?;
    if unsafe { flock(f.as_raw_fd(), LOCK_EX | LOCK_NB) } != 0 {
        return Err(EngineError::AlreadyRunning(path.display().to_string()));
    }
    Ok(f)
}

fn load_generator(inner: &Inner) -> Result<Generator, String> {
    inner.set_status(Status::Loading { detail: "loading Ministral-3-8B-Instruct-2512 (4-bit) on MLX".into() });
    match std::panic::catch_unwind(|| Generator::load(&inner.model_dir)) {
        Ok(Ok(g)) => { *inner.identity.lock().unwrap() = Some(g.identity.clone()); inner.model_loaded.store(true, Ordering::Relaxed); inner.set_status(Status::Ready); Ok(g) }
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(frost_mlx::take_last_error().unwrap_or_else(|| "generator crashed while loading".into())),
    }
}

fn worker(inner: Arc<Inner>, rx: Receiver<Cmd>) {
    let mut gen = match load_generator(&inner) {
        Ok(g) => Some(g),
        Err(detail) => { inner.set_status(Status::Error { detail }); return drain_unavailable(&inner, rx); }
    };
    loop {
        // Idle policy: after IDLE_UNLOAD without a request, release the weights (the OS would
        // otherwise compress/swap them silently) and reload on the next message.
        let cmd = match rx.recv_timeout(IDLE_UNLOAD) {
            Ok(c) => c,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if gen.take().is_some() {
                    frost_mlx::clear_cache();
                    inner.kv_bytes.store(0, Ordering::Relaxed);
                    inner.model_loaded.store(false, Ordering::Relaxed);
                    inner.emit(&Event::Status { status: Status::Ready });
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let conv_id = match &cmd { Cmd::Shutdown => break, Cmd::Generate { conv_id } | Cmd::Execute { conv_id, .. } => conv_id.clone() };
        // The model may have been unloaded (memory pressure or idle): bring it back for this request.
        if gen.is_none() {
            match load_generator(&inner) {
                Ok(g) => gen = Some(g),
                Err(detail) => {
                    inner.set_status(Status::Ready); // the engine is alive; this request failed to get a model
                    inner.emit(&Event::Error { conv_id: Some(conv_id), detail: format!("could not reload the model: {detail}") });
                    inner.done();
                    continue;
                }
            }
        }
        let g = gen.as_mut().expect("loaded");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &cmd {
            Cmd::Execute { conv_id, attempt_id } => { if run_execute(&inner, conv_id, attempt_id) { run_generation(&inner, g, conv_id); } }
            Cmd::Generate { conv_id } => run_generation(&inner, g, conv_id),
            Cmd::Shutdown => {}
        }));
        if r.is_err() {
            let detail = frost_mlx::take_last_error().unwrap_or_else(|| "generation crashed".into());
            g.reset(); // the graph/cache state is unknown after a panic: drop it
            inner.emit(&Event::Error { conv_id: Some(conv_id.clone()), detail: format!("generation failed: {detail}") });
        }
        inner.kv_bytes.store(g.kv_cache_bytes(), Ordering::Relaxed);
        // Sustained critical memory pressure: give the 4.5 GB back rather than let the OS swap the
        // weights; the next request reloads (visible as a loading status, never as a hidden stall).
        if inner.unload_requested.swap(false, Ordering::Relaxed) {
            gen = None;
            frost_mlx::clear_cache();
            inner.kv_bytes.store(0, Ordering::Relaxed);
            inner.model_loaded.store(false, Ordering::Relaxed);
            inner.emit(&Event::Note { conv_id: conv_id.clone(), text: "model unloaded under critical memory pressure; it reloads automatically on your next message".into() });
        }
        inner.done();
    }
    // free the model before the thread exits and hand MLX's cached buffers back to the OS
    drop(gen);
    frost_mlx::clear_cache();
}

fn drain_unavailable(inner: &Inner, rx: Receiver<Cmd>) {
    for cmd in rx {
        match cmd {
            Cmd::Shutdown => break,
            Cmd::Generate { conv_id } | Cmd::Execute { conv_id, .. } => {
                inner.emit(&Event::Error { conv_id: Some(conv_id), detail: "model unavailable".into() });
                inner.done();
            }
        }
    }
}

/// Which stored messages the model sees: everything after the last "context cleared" marker.
fn context_slice(msgs: &[Message]) -> &[Message] {
    match msgs.iter().rposition(|m| m.role == Role::System) { Some(i) => &msgs[i + 1..], None => msgs }
}

fn to_gen_messages(msgs: &[Message]) -> Vec<frost_gen::Message> {
    msgs.iter().filter_map(|m| {
        let role = match m.role { Role::User => frost_gen::Role::User, Role::Assistant => frost_gen::Role::Assistant, Role::Tool => frost_gen::Role::Tool, Role::System => return None };
        let calls: Vec<ToolCall> = serde_json::from_value(m.meta["tool_calls"].clone()).unwrap_or_default();
        if m.content.is_empty() && calls.is_empty() { return None; }
        Some(frost_gen::Message::new(role, m.content.clone()).with_tool_calls(calls))
    }).collect()
}

/// Run an approved attempt on the worker, persist the outcome, feed it back as a tool turn.
/// Returns true when the model should now continue (a generation follows).
fn run_execute(inner: &Inner, conv_id: &str, attempt_id: &str) -> bool {
    let (attempt, repo) = {
        let st = inner.store.lock().unwrap();
        let a = st.list_attempts(conv_id).ok().and_then(|v| v.into_iter().find(|a| a.id == attempt_id));
        let repo = st.get_conversation(conv_id).ok().and_then(|c| c.repo_path);
        (a, repo)
    };
    let (Some(attempt), Some(repo)) = (attempt, repo) else {
        inner.emit(&Event::Error { conv_id: Some(conv_id.into()), detail: "attempt or repository missing".into() });
        return false;
    };
    let (status, exit_code, stdout, stderr, result) = agent::execute(Path::new(&repo), &attempt, &inner.cancel);
    {
        let mut st = inner.store.lock().unwrap();
        let _ = st.finish_attempt(attempt_id, status, exit_code, &stdout, &stderr, None);
        let _ = st.append_message(conv_id, Role::Tool, &result, json!({"attempt_id": attempt_id, "tool": attempt.kind.as_str(), "status": status.as_str()}));
    }
    inner.emit(&Event::AttemptUpdated { conv_id: conv_id.into(), attempt: attempt_json(&attempt, status) });
    inner.emit(&Event::MessagesChanged { conv_id: conv_id.into() });
    if status == AttemptStatus::Cancelled { return false; }
    let used = agent::repairs_used(&inner.store.lock().unwrap(), conv_id);
    if used >= agent::MAX_REPAIRS {
        inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("repair budget exhausted: {used} approved test runs failed; FROST will summarize instead of proposing more changes") });
    }
    true
}

fn run_generation(inner: &Inner, gen: &mut Generator, conv_id: &str) {
    let (conv, msgs) = {
        let st = inner.store.lock().unwrap();
        match (st.get_conversation(conv_id), st.list_messages(conv_id)) {
            (Ok(c), Ok(m)) => (c, m),
            (Err(e), _) | (_, Err(e)) => { inner.emit(&Event::Error { conv_id: Some(conv_id.into()), detail: e.to_string() }); return; }
        }
    };
    let ctx = context_slice(&msgs);
    let mut history = to_gen_messages(ctx);
    if history.last().map(|m| m.role == frost_gen::Role::Assistant).unwrap_or(true) {
        inner.emit(&Event::Error { conv_id: Some(conv_id.into()), detail: "nothing to answer: the last message is not from the user".into() });
        return;
    }
    let repo = conv.repo_path.as_deref().filter(|p| Path::new(p).is_dir()).map(PathBuf::from);
    let mut system = if conv.system_prompt.is_empty() { DEFAULT_SYSTEM_PROMPT.to_string() } else { conv.system_prompt.clone() };
    let tools = if repo.is_some() { system.push_str(agent::TOOL_GUIDE); Some(agent::tools_json()) } else { None };
    let mode = *inner.mode.lock().unwrap();

    // Repository memory: retrieve source excerpts for the user's latest message and prepend them,
    // framed as data, to the model-facing copy of that turn (the stored message is untouched).
    let mut retrieved: Option<(frost_repo::ContextBlock, Vec<frost_repo::Hit>)> = None;
    if let (Some(root), Some(last)) = (&repo, history.last_mut()) {
        if last.role == frost_gen::Role::User {
            retrieved = retrieve(inner, conv_id, root, &last.content, mode);
            if let Some((cb, _)) = &retrieved { last.content = format!("{}\n{}", cb.text, last.content); }
        }
    }

    // Context budget: drop the oldest turns until the prompt fits; never silently.
    let limit = CONTEXT_BUDGET - RESERVED_OUTPUT_TOKENS;
    let mut dropped = 0usize;
    let prompt = loop {
        let ids = match frost_gen::render(gen.tokenizer(), &system, tools.as_deref(), &history) {
            Ok(ids) => ids,
            Err(e) => { inner.emit(&Event::Error { conv_id: Some(conv_id.into()), detail: e.to_string() }); return; }
        };
        if ids.len() <= limit { break ids; }
        if history.len() <= 1 {
            inner.emit(&Event::Error { conv_id: Some(conv_id.into()), detail: format!("message too long: {} tokens; at most {limit} fit in the model's context", ids.len()) });
            return;
        }
        history.remove(0);
        dropped += 1;
    };
    if dropped > 0 {
        inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("context truncated: the {dropped} oldest message(s) were left out of the model's context (still in the transcript)") });
    }

    // Admission: no new uncached neural work while the OS reports Serious/Critical thermal state
    // (or critical memory pressure). Deferred visibly; Regenerate retries once it recovers.
    let t_now = inner.thermal();
    let p_now = frost_platform::memory_pressure();
    if matches!(t_now, ThermalState::Serious | ThermalState::Critical) || p_now == frost_platform::MemoryPressure::Critical {
        let why = if p_now == frost_platform::MemoryPressure::Critical { "critical memory pressure".to_string() } else { format!("thermal state {t_now:?}") };
        let meta = json!({"status": "done", "finish": if p_now == frost_platform::MemoryPressure::Critical { "memory_pressure_deferred".to_string() } else { format!("thermal_deferred:{t_now:?}") }, "deferred_before_start": true, "mode": mode_name(mode)});
        if let Ok(m) = inner.store.lock().unwrap().append_message(conv_id, Role::Assistant, "", meta.clone()) {
            inner.emit(&Event::MessageDone { conv_id: conv_id.into(), message_id: m.id, content: String::new(), meta });
        }
        inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("deferred: {why}; FROST admits no new work until the system recovers. Use Regenerate to retry.") });
        inner.emit(&Event::MessagesChanged { conv_id: conv_id.into() });
        return;
    }

    let message_id = {
        let mut st = inner.store.lock().unwrap();
        match st.append_message(conv_id, Role::Assistant, "", json!({"status": "generating"})) {
            Ok(m) => m.id,
            Err(e) => { inner.emit(&Event::Error { conv_id: Some(conv_id.into()), detail: e.to_string() }); return; }
        }
    };
    inner.emit(&Event::MessageStarted { conv_id: conv_id.into(), message_id: message_id.clone() });

    let params = GenParams {
        max_new_tokens: RESERVED_OUTPUT_TOKENS,
        sample: SampleParams::default(),
        prefill_chunk: match mode { Mode::Quiet => 256, Mode::Balanced => 512, Mode::Performance => 1024 },
    };
    let mut pending = String::new();
    let mut last_flush = Instant::now();
    let mut last_thermal = Instant::now();
    let mut thermal_stop: Option<ThermalState> = None;
    let mut pressure_stop = false;
    let mut pressure_warn = false;
    let mut in_tool_call = false;
    let mut step_t = Instant::now();
    let t_start = Instant::now();
    let flush = |pending: &mut String| {
        if !pending.is_empty() {
            inner.emit(&Event::Delta { conv_id: conv_id.into(), message_id: message_id.clone(), text: std::mem::take(pending) });
        }
    };
    // Governor poll, shared by prefill chunks and decode steps: re-read the real OS thermal state
    // and memory pressure; Serious/Critical (or critical pressure) stops at this safe boundary.
    // There is no production override.
    let mut governor = |force: bool| {
        if !force && last_thermal.elapsed() < THERMAL_POLL { return; }
        last_thermal = Instant::now();
        let t = inner.thermal();
        if matches!(t, ThermalState::Serious | ThermalState::Critical) { thermal_stop = Some(t); inner.cancel.store(true, Ordering::Relaxed); }
        // Memory pressure: warn -> release MLX's buffer cache after this reply; critical ->
        // stop now and drop the KV cache rather than let the OS swap the weights.
        match frost_platform::memory_pressure() {
            frost_platform::MemoryPressure::Critical => { pressure_stop = true; inner.cancel.store(true, Ordering::Relaxed); }
            frost_platform::MemoryPressure::Warn => pressure_warn = true,
            frost_platform::MemoryPressure::Normal => {}
        }
    };
    let result = gen.generate(&prompt, &params, &inner.cancel, &mut |e| match e {
        GenEvent::PrefillChunk { .. } => governor(true),
        GenEvent::Prefilled { .. } => {}
        GenEvent::Token { id, text } => {
            // Tool-call bytes are structured output, not prose: keep them out of the streamed transcript.
            if id == ctl::TOOL_CALLS { in_tool_call = true; }
            if !in_tool_call { pending.push_str(text); }
            if last_flush.elapsed() >= DELTA_FLUSH { flush(&mut pending); last_flush = Instant::now(); }
            governor(false);
            // Quiet mode: restrained duty cycle — rest 0.4x each decode step's compute time
            // (about 29% of wall time), capped at 60 ms.
            if mode == Mode::Quiet {
                let step = step_t.elapsed();
                std::thread::sleep(step.mul_f32(0.4).min(Duration::from_millis(60)));
            }
            step_t = Instant::now();
        }
    });
    flush(&mut pending);
    if pressure_stop { gen.reset(); frost_mlx::clear_cache(); inner.unload_requested.store(true, Ordering::Relaxed); }
    else if pressure_warn { frost_mlx::clear_cache(); }

    match result {
        Ok((ids, stats)) => {
            let (text, calls) = frost_gen::parse_output(gen.tokenizer(), &ids);
            let finish = match (thermal_stop, stats.finish) {
                _ if pressure_stop => "memory_pressure_deferred".to_string(),
                (Some(t), _) => format!("thermal_deferred:{t:?}"),
                (None, FinishReason::Eos) => "eos".into(),
                (None, FinishReason::MaxTokens) => "max_tokens".into(),
                (None, FinishReason::Cancelled) => "cancelled".into(),
            };
            let meta = json!({
                "status": "done", "finish": finish, "mode": mode_name(mode),
                "generator": { "repo": gen.identity.repo, "revision": gen.identity.revision, "quantization": gen.identity.quantization, "backend": gen.identity.backend },
                "sampler": stats.sampler, "prompt_tokens": stats.prompt_tokens, "reused_prefix_tokens": stats.reused_prefix_tokens,
                "new_tokens": stats.new_tokens, "prefill_ms": stats.prefill_ms, "decode_ms": stats.decode_ms, "tokens_per_second": stats.tokens_per_second,
                "wall_ms": t_start.elapsed().as_secs_f64() * 1e3, "mlx_peak_bytes": stats.mlx_peak_bytes, "kv_cache_bytes": stats.kv_cache_bytes,
                "context_truncated_messages": dropped,
                "tools_available": tools.is_some(),
                "tool_calls": calls,
                // which source spans entered the prompt, re-verified against the files right now
                "citations": retrieved.as_ref().map(|(cb, hits)| {
                    let guard = inner.indexer.lock().unwrap();
                    cb.citations.iter().zip(hits.iter()).map(|(c, h)| json!({
                        "n": c.n, "path": c.path, "start_line": c.start_line, "end_line": c.end_line, "digest": c.digest_hex, "chunk_id": c.chunk_id,
                        "status": guard.as_ref().and_then(|ix| repo.as_deref().map(|r| ix.verify_citation(r, h))).map(|s| format!("{s:?}")).unwrap_or_else(|| "unverified".into()),
                        "vector_rank": h.rank_vector, "lexical_rank": h.rank_lexical,
                    })).collect::<Vec<_>>()
                }).unwrap_or_default(),
                "control_ids_seen": ids.iter().filter(|&&i| i < ctl::FIRST_TEXT_ID).count(),
            });
            if let Err(e) = inner.store.lock().unwrap().update_message(&message_id, &text, meta.clone()) {
                inner.emit(&Event::Error { conv_id: Some(conv_id.into()), detail: e.to_string() });
            }
            if let Some(t) = thermal_stop {
                inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("paused: the system reported thermal state {t:?}; FROST stops new work until it recovers. Use Regenerate to continue later.") });
            }
            if pressure_stop {
                inner.emit(&Event::Note { conv_id: conv_id.into(), text: "paused: the system reported critical memory pressure; FROST stopped and released its KV cache instead of swapping. Close other apps and use Regenerate.".into() });
            }
            inner.emit(&Event::MessageDone { conv_id: conv_id.into(), message_id: message_id.clone(), content: text, meta });
            inner.emit(&Event::ConversationsChanged);

            // Coding loop: dispatch tool calls (only with an attached repository and a clean finish).
            if let (Some(root), false) = (&repo, calls.is_empty()) {
                if finish != "eos" { return; }
                let rounds = agent::tool_rounds_used(&inner.store.lock().unwrap(), conv_id);
                if rounds > agent::MAX_TOOL_ROUNDS {
                    inner.emit(&Event::Note { conv_id: conv_id.into(), text: format!("stopped: {} tool rounds in one turn is the limit; send a new message to continue", agent::MAX_TOOL_ROUNDS) });
                    return;
                }
                let mut proposed = false;
                for call in &calls {
                    match agent::dispatch(&inner.store, root, conv_id, &message_id, call) {
                        agent::Outcome::Result(r) => {
                            let _ = inner.store.lock().unwrap().append_message(conv_id, Role::Tool, &r, json!({"tool": call.name}));
                        }
                        agent::Outcome::Proposed(a) => {
                            proposed = true;
                            inner.emit(&Event::AttemptProposed { conv_id: conv_id.into(), attempt: attempt_json(&a, AttemptStatus::Proposed) });
                        }
                    }
                }
                inner.emit(&Event::MessagesChanged { conv_id: conv_id.into() });
                if !proposed {
                    // read-only results appended: let the model continue (the worker loop picks this up next)
                    let _ = inner.enqueue(Cmd::Generate { conv_id: conv_id.into() });
                }
            }
        }
        Err(e) => {
            let detail = e.to_string();
            let _ = inner.store.lock().unwrap().update_message(&message_id, "", json!({"status": "error", "detail": detail}));
            inner.emit(&Event::Error { conv_id: Some(conv_id.into()), detail });
            inner.emit(&Event::MessagesChanged { conv_id: conv_id.into() });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("frost-chat-{tag}-{n}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn context_slice_stops_at_last_clear_marker() {
        let mk = |role: Role, seq: i64| Message { id: format!("m{seq}"), conversation_id: "c".into(), seq, role, content: "x".into(), created_at: 0, meta: json!({}) };
        let msgs = vec![mk(Role::User, 1), mk(Role::Assistant, 2), mk(Role::System, 3), mk(Role::User, 4)];
        assert_eq!(context_slice(&msgs).len(), 1);
        assert_eq!(to_gen_messages(&msgs).len(), 3);
    }

    #[test]
    fn engine_without_model_reports_error_status_and_refuses_sends() {
        // Point the engine at an empty data dir: the store works, the generator is unavailable.
        let dir = tmp_dir("nomodel");
        let eng = Engine::new(&dir).unwrap();
        assert!(eng.wait_idle(Duration::from_secs(20)));
        assert!(matches!(eng.status(), Status::Error { .. }), "{:?}", eng.status());
        let c = eng.create_conversation().unwrap();
        assert!(matches!(eng.send(&c.id, "hi"), Err(EngineError::Unavailable(_))));
        assert!(matches!(eng.send(&c.id, "   "), Err(EngineError::Empty)));
        eng.set_mode("balanced").unwrap();
        assert_eq!(eng.mode(), Mode::Balanced);
        assert!(matches!(eng.set_mode("turbo"), Err(EngineError::BadMode(_))));
        // single-instance: a second engine on the same data dir is refused
        assert!(matches!(Engine::new(&dir), Err(EngineError::AlreadyRunning(_))));
        eng.shutdown();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore = "requires the pinned checkpoint under ~/Library/Application Support/FROST"]
    fn streams_persists_regenerates_cancels_and_honours_thermal() {
        let data = PathBuf::from(std::env::var("HOME").unwrap()).join("Library/Application Support/FROST");
        let dir = tmp_dir("engine");
        // real models, throwaway database: symlink the models dir into the temp data dir
        std::os::unix::fs::symlink(data.join("models"), dir.join("models")).unwrap();
        let eng = Engine::new(&dir).unwrap();
        let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
        { let ev = events.clone(); eng.subscribe(Box::new(move |e| ev.lock().unwrap().push(e.clone()))); }
        assert!(eng.wait_idle(Duration::from_secs(120)), "load timed out");
        assert_eq!(eng.status(), Status::Ready);

        let c = eng.create_conversation().unwrap();
        eng.send(&c.id, "Reply with exactly one word: the capital of France.").unwrap();
        assert!(matches!(eng.send(&c.id, "again"), Err(EngineError::Busy)));
        assert!(eng.wait_idle(Duration::from_secs(60)));
        let msgs = eng.messages(&c.id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert!(msgs[1].content.to_lowercase().contains("paris"), "{:?}", msgs[1].content);
        assert_eq!(msgs[1].meta["status"], "done");
        assert_eq!(msgs[1].meta["generator"]["repo"], frost_gen::REPO);
        assert!(events.lock().unwrap().iter().any(|e| matches!(e, Event::Delta { .. })));
        assert!(events.lock().unwrap().iter().any(|e| matches!(e, Event::MessageDone { .. })));

        // regenerate replaces the assistant turn
        eng.regenerate(&c.id).unwrap();
        assert!(eng.wait_idle(Duration::from_secs(60)));
        let msgs = eng.messages(&c.id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert!(msgs[1].meta["reused_prefix_tokens"].as_u64().unwrap() > 50, "prefix reuse across regenerate: {}", msgs[1].meta);

        // cancellation leaves a consistent partial message and the next request works
        eng.send(&c.id, "Count from 1 to 300, one number per line.").unwrap();
        std::thread::sleep(Duration::from_millis(1500));
        eng.cancel();
        assert!(eng.wait_idle(Duration::from_secs(60)));
        let msgs = eng.messages(&c.id).unwrap();
        assert_eq!(msgs.last().unwrap().meta["finish"], "cancelled");
        eng.send(&c.id, "Say hello in one word.").unwrap();
        assert!(eng.wait_idle(Duration::from_secs(60)));
        assert_eq!(eng.messages(&c.id).unwrap().last().unwrap().meta["finish"], "eos");

        // simulated serious thermal state stops the generation at a token boundary
        eng.simulate_thermal(Some(ThermalState::Serious));
        eng.send(&c.id, "Count from 1 to 300, one number per line.").unwrap();
        assert!(eng.wait_idle(Duration::from_secs(60)));
        let last = eng.messages(&c.id).unwrap().last().unwrap().clone();
        assert!(last.meta["finish"].as_str().unwrap().starts_with("thermal_deferred"), "{}", last.meta);
        eng.simulate_thermal(None);

        // clear context: the model only sees messages after the marker
        eng.clear_context(&c.id).unwrap();
        eng.send(&c.id, "What did I ask you first? Answer in one short sentence.").unwrap();
        assert!(eng.wait_idle(Duration::from_secs(60)));
        let msgs = eng.messages(&c.id).unwrap();
        assert!(msgs.last().unwrap().meta["prompt_tokens"].as_u64().unwrap() < 200, "{}", msgs.last().unwrap().meta);

        eng.shutdown();
        let _ = std::fs::remove_dir_all(dir);
    }

    /// §12 item 4: repository-grounded answers with citations that match the source bytes, and a
    /// changed file invalidating the cached context (re-index + fresh citations).
    #[test]
    #[ignore = "requires both pinned models under ~/Library/Application Support/FROST"]
    fn repository_grounded_answers_cite_verified_spans_and_track_file_changes() {
        let data = PathBuf::from(std::env::var("HOME").unwrap()).join("Library/Application Support/FROST");
        let dir = tmp_dir("repo");
        std::os::unix::fs::symlink(data.join("models"), dir.join("models")).unwrap();
        // a private copy of the rust fixture as the attached repository
        let repo = dir.join("repo");
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/rust-slugify");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::create_dir_all(repo.join("tests")).unwrap();
        for f in ["Cargo.toml", "src/lib.rs", "tests/held_out.rs", "TASK.md"] { std::fs::copy(src.join(f), repo.join(f)).unwrap(); }
        let eng = Engine::new(&dir).unwrap();
        let notes: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        { let n = notes.clone(); eng.subscribe(Box::new(move |e| if let Event::Note { text, .. } = e { n.lock().unwrap().push(text.clone()) })); }
        assert!(eng.wait_idle(Duration::from_secs(120)));
        let c = eng.create_conversation().unwrap();
        eng.set_repo(&c.id, Some(&repo.display().to_string())).unwrap();
        eng.send(&c.id, "Which function in this repository turns text into a URL slug, and in which file is it? Answer in one sentence citing the file.").unwrap();
        assert!(eng.wait_idle(Duration::from_secs(180)));
        let reply = eng.messages(&c.id).unwrap().into_iter().filter(|m| m.role == Role::Assistant).last().unwrap();
        let cites = reply.meta["citations"].as_array().cloned().unwrap_or_default();
        assert!(!cites.is_empty(), "no citations: {}", reply.meta);
        assert!(cites.iter().all(|c| c["status"] == "Verified"), "citations must verify against the bytes: {cites:?}");
        assert!(cites.iter().any(|c| c["path"] == "src/lib.rs"), "{cites:?}");
        assert!(reply.content.to_lowercase().contains("slugify"), "grounded answer should name slugify: {}", reply.content);
        assert!(notes.lock().unwrap().iter().any(|n| n.starts_with("repository indexed")), "{:?}", notes.lock().unwrap());
        assert_eq!(reply.meta["tools_available"], true);

        // change the source: the next turn must re-index and cite the NEW bytes
        let mut lib = std::fs::read_to_string(repo.join("src/lib.rs")).unwrap();
        lib = lib.replace("pub fn slugify", "/// Produces a slug.\npub fn make_url_slug");
        std::fs::write(repo.join("src/lib.rs"), lib).unwrap();
        notes.lock().unwrap().clear();
        eng.send(&c.id, "What is the slug function called now? One sentence.").unwrap();
        assert!(eng.wait_idle(Duration::from_secs(180)));
        let reply2 = eng.messages(&c.id).unwrap().into_iter().filter(|m| m.role == Role::Assistant).last().unwrap();
        let cites2 = reply2.meta["citations"].as_array().cloned().unwrap_or_default();
        assert!(notes.lock().unwrap().iter().any(|n| n.starts_with("repository indexed")), "changed file must trigger a refresh: {:?}", notes.lock().unwrap());
        assert!(cites2.iter().all(|c| c["status"] == "Verified"), "{cites2:?}");
        assert!(cites2.iter().any(|c| c["path"] == "src/lib.rs" && c["digest"] != cites.iter().find(|o| o["path"] == "src/lib.rs").map(|o| o["digest"].clone()).unwrap_or_default()),
            "the cited chunk digest must change with the file: {cites2:?}");
        eng.shutdown();
        let _ = std::fs::remove_dir_all(dir);
    }
}

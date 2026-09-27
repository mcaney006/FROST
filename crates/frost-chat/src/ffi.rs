//! C ABI for the native (Objective-C/AppKit) shell. Mirrors `native/app/frost_app.h`.
//!
//! Rules at this boundary: no Rust panic crosses it (everything is wrapped in
//! `catch_unwind`), every returned `char*` is owned by Rust and must be released with
//! `frost_string_free`, every `int32_t` return is 0 on success and negative on failure
//! (`frost_last_error` has the detail), and the event callback fires on a background
//! thread — the UI must hop to its main thread itself. A handle whose engine could not be
//! created (e.g. the single-instance lock is held) still answers every call: status reports
//! `error` with the reason and actions fail with that reason.

use crate::{Engine, Event};
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Mutex;

pub struct FrostEngine {
    engine: Result<Engine, String>,
    last_error: Mutex<String>,
}

pub type FrostEventCb = Option<extern "C" fn(ctx: *mut c_void, json_utf8: *const c_char)>;

struct CbCtx(*mut c_void);
impl CbCtx { fn ptr(&self) -> *mut c_void { self.0 } }
// SAFETY: the pointer is an opaque token the C side gave us and only ever hands back to its
// own callback; the UI is responsible for its lifetime and thread-safety.
unsafe impl Send for CbCtx {}
unsafe impl Sync for CbCtx {}

fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() { return None; }
    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}
fn out(s: String) -> *mut c_char {
    CString::new(s.replace('\0', "")).map(|c| c.into_raw()).unwrap_or(std::ptr::null_mut())
}
fn with<T>(h: *mut FrostEngine, f: impl FnOnce(&Engine) -> Result<T, String>, on_err: T) -> T {
    if h.is_null() { return on_err; }
    let e = unsafe { &*h };
    let r = catch_unwind(AssertUnwindSafe(|| match &e.engine { Ok(eng) => f(eng), Err(why) => Err(why.clone()) }));
    match r {
        Ok(Ok(v)) => v,
        Ok(Err(msg)) => { if let Ok(mut l) = e.last_error.lock() { *l = msg; } on_err }
        Err(_) => { if let Ok(mut l) = e.last_error.lock() { *l = "internal error (panic caught at the FFI boundary)".into(); } on_err }
    }
}
fn status_of<T>(h: *mut FrostEngine, f: impl FnOnce(&Engine) -> Result<T, crate::EngineError>) -> i32 {
    with(h, |e| f(e).map(|_| 0).map_err(|err| err.to_string()), -1)
}
fn json_of(h: *mut FrostEngine, f: impl FnOnce(&Engine) -> Result<serde_json::Value, crate::EngineError>) -> *mut c_char {
    with(h, |e| f(e).map(|v| out(v.to_string())).map_err(|err| err.to_string()), std::ptr::null_mut())
}

/// Create the engine for a data directory (Application Support/FROST). Never returns NULL and
/// never panics: a failed creation yields a handle whose status is `error`.
#[no_mangle]
pub extern "C" fn frost_engine_new(data_dir_utf8: *const c_char) -> *mut FrostEngine {
    let dir = cstr(data_dir_utf8).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Library/Application Support/FROST"));
    let engine = match catch_unwind(|| Engine::new(&dir)) {
        Ok(Ok(e)) => Ok(e),
        Ok(Err(err)) => Err(err.to_string()),
        Err(_) => Err("engine crashed while starting (panic caught at the FFI boundary)".to_string()),
    };
    Box::into_raw(Box::new(FrostEngine { engine, last_error: Mutex::new(String::new()) }))
}

/// Cancel, stop the worker, release the model, free the handle.
#[no_mangle]
pub extern "C" fn frost_engine_free(h: *mut FrostEngine) {
    if h.is_null() { return; }
    let b = unsafe { Box::from_raw(h) };
    if let Ok(e) = &b.engine { let _ = catch_unwind(AssertUnwindSafe(|| e.shutdown())); }
}

/// Register the event callback (JSON events, background thread). NULL unregisters all callbacks.
#[no_mangle]
pub extern "C" fn frost_engine_set_event_callback(h: *mut FrostEngine, cb: FrostEventCb, ctx: *mut c_void) {
    let Some(cb) = cb else {
        let _ = status_of(h, |e| { e.clear_subscribers(); Ok(()) });
        return;
    };
    let ctx = CbCtx(ctx);
    let _ = status_of(h, |e| {
        e.clear_subscribers();
        e.subscribe(Box::new(move |ev: &Event| {
            if let Ok(js) = serde_json::to_string(ev) {
                if let Ok(c) = CString::new(js) { cb(ctx.ptr(), c.as_ptr()); }
            }
        }));
        Ok(())
    });
}

/// Status JSON. For a handle without an engine: {"state":"error","detail":<reason>,"generating":false}.
#[no_mangle]
pub extern "C" fn frost_engine_status_json(h: *mut FrostEngine) -> *mut c_char {
    if h.is_null() { return out(r#"{"state":"error","detail":"null engine","generating":false}"#.into()); }
    let e = unsafe { &*h };
    match &e.engine {
        Ok(_) => json_of(h, |e| Ok(e.status_json())),
        Err(why) => out(serde_json::json!({"state": "error", "detail": why, "generating": false, "mode": "quiet"}).to_string()),
    }
}
#[no_mangle] pub extern "C" fn frost_model_info_json(h: *mut FrostEngine) -> *mut c_char { json_of(h, |e| Ok(e.model_info())) }
#[no_mangle] pub extern "C" fn frost_conversations_json(h: *mut FrostEngine) -> *mut c_char { json_of(h, |e| Ok(serde_json::to_value(e.conversations()?).unwrap_or_default())) }
#[no_mangle] pub extern "C" fn frost_messages_json(h: *mut FrostEngine, conv_id: *const c_char) -> *mut c_char {
    let id = cstr(conv_id).unwrap_or_default();
    json_of(h, |e| Ok(serde_json::to_value(e.messages(&id)?).unwrap_or_default()))
}
/// Returns the new conversation id (bare string, not JSON).
#[no_mangle] pub extern "C" fn frost_conversation_create(h: *mut FrostEngine) -> *mut c_char {
    with(h, |e| e.create_conversation().map(|c| out(c.id)).map_err(|err| err.to_string()), std::ptr::null_mut())
}
#[no_mangle] pub extern "C" fn frost_conversation_delete(h: *mut FrostEngine, id: *const c_char) -> i32 { let id = cstr(id).unwrap_or_default(); status_of(h, |e| e.delete_conversation(&id)) }
#[no_mangle] pub extern "C" fn frost_conversation_rename(h: *mut FrostEngine, id: *const c_char, title: *const c_char) -> i32 {
    let (id, t) = (cstr(id).unwrap_or_default(), cstr(title).unwrap_or_default());
    status_of(h, |e| e.rename_conversation(&id, &t))
}
#[no_mangle] pub extern "C" fn frost_send(h: *mut FrostEngine, conv_id: *const c_char, text: *const c_char) -> i32 {
    let (id, t) = (cstr(conv_id).unwrap_or_default(), cstr(text).unwrap_or_default());
    status_of(h, |e| e.send(&id, &t))
}
#[no_mangle] pub extern "C" fn frost_regenerate(h: *mut FrostEngine, conv_id: *const c_char) -> i32 { let id = cstr(conv_id).unwrap_or_default(); status_of(h, |e| e.regenerate(&id)) }
#[no_mangle] pub extern "C" fn frost_clear_context(h: *mut FrostEngine, conv_id: *const c_char) -> i32 { let id = cstr(conv_id).unwrap_or_default(); status_of(h, |e| e.clear_context(&id)) }
#[no_mangle] pub extern "C" fn frost_cancel(h: *mut FrostEngine) -> i32 { status_of(h, |e| { e.cancel(); Ok(()) }) }
#[no_mangle] pub extern "C" fn frost_set_mode(h: *mut FrostEngine, mode: *const c_char) -> i32 { let m = cstr(mode).unwrap_or_default(); status_of(h, |e| e.set_mode(&m)) }
/// `path` may be NULL to detach the repository.
#[no_mangle] pub extern "C" fn frost_set_repo(h: *mut FrostEngine, conv_id: *const c_char, path: *const c_char) -> i32 {
    let (id, p) = (cstr(conv_id).unwrap_or_default(), cstr(path));
    status_of(h, |e| e.set_repo(&id, p.as_deref()))
}

// ---------------------------------------------------------------------------------------------
// coding-agent attempts (patches / commands awaiting or holding a user decision)

/// [{"id","conversation_id","message_id","kind":"propose_diff"|"run_command","payload":{...},"status","exit_code","stdout","stderr","duration_ms","created_at","updated_at"}]
#[no_mangle] pub extern "C" fn frost_attempts_json(h: *mut FrostEngine, conv_id: *const c_char) -> *mut c_char {
    let id = cstr(conv_id).unwrap_or_default();
    json_of(h, |e| Ok(serde_json::to_value(e.attempts(&id)?).unwrap_or_default()))
}
/// approve != 0 executes the attempt on the worker thread; 0 denies it. Either way the model is told.
#[no_mangle] pub extern "C" fn frost_attempt_decide(h: *mut FrostEngine, conv_id: *const c_char, attempt_id: *const c_char, approve: i32) -> i32 {
    let (c, a) = (cstr(conv_id).unwrap_or_default(), cstr(attempt_id).unwrap_or_default());
    status_of(h, |e| e.decide_attempt(&c, &a, approve != 0))
}

/// Detail of the most recent failed call on this handle (or the engine-creation failure).
#[no_mangle] pub extern "C" fn frost_last_error(h: *mut FrostEngine) -> *mut c_char {
    if h.is_null() { return out("null engine".into()); }
    let e = unsafe { &*h };
    let last = e.last_error.lock().map(|s| s.clone()).unwrap_or_default();
    if last.is_empty() { if let Err(why) = &e.engine { return out(why.clone()); } }
    out(last)
}
#[no_mangle] pub extern "C" fn frost_string_free(s: *mut c_char) { if !s.is_null() { unsafe { drop(CString::from_raw(s)); } } }

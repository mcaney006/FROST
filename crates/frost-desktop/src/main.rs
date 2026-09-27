//! FROST desktop: Rust owns the engine lifecycle; the window is native AppKit (native/app/*.m)
//! talking to the engine over the C ABI in native/app/frost_app.h.
use frost_chat::ffi::{self, FrostEngine};
use std::ffi::{c_char, c_int, CStr, CString};
use std::time::{Duration, Instant};

#[allow(improper_ctypes)] // FrostEngine is only ever handled through an opaque pointer
extern "C" {
    fn frost_ui_run(engine: *mut FrostEngine, argc: c_int, argv: *const *const c_char) -> c_int;
}

/// Copy a Rust-owned ABI string out and release it.
fn take(p: *mut c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    ffi::frost_string_free(p);
    s
}

/// `--headless-check`: load the engine, wait for it to leave `loading` (≤120 s), print status +
/// model identity, exit 0 when ready / 3 on error. No UI.
fn headless_check() -> i32 {
    let e = ffi::frost_engine_new(std::ptr::null());
    let started = Instant::now();
    let code = loop {
        let status = take(ffi::frost_engine_status_json(e));
        let state = serde_json::from_str::<serde_json::Value>(&status)
            .ok()
            .and_then(|v| v["state"].as_str().map(str::to_owned))
            .unwrap_or_default();
        if state != "loading" || started.elapsed() > Duration::from_secs(120) {
            println!("{status}");
            println!("{}", take(ffi::frost_model_info_json(e)));
            break if state == "ready" { 0 } else { 3 };
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    ffi::frost_engine_free(e);
    code
}

fn main() {
    // The Objective-C objects resolve these by name at link time; referencing them here keeps
    // the linker from dead-stripping the exported ABI out of the frost-chat rlib.
    let keep: [*const (); 20] = [
        ffi::frost_attempts_json as *const (),
        ffi::frost_attempt_decide as *const (),
        ffi::frost_engine_new as *const (),
        ffi::frost_engine_free as *const (),
        ffi::frost_engine_set_event_callback as *const (),
        ffi::frost_engine_status_json as *const (),
        ffi::frost_model_info_json as *const (),
        ffi::frost_conversations_json as *const (),
        ffi::frost_conversation_create as *const (),
        ffi::frost_conversation_delete as *const (),
        ffi::frost_conversation_rename as *const (),
        ffi::frost_messages_json as *const (),
        ffi::frost_send as *const (),
        ffi::frost_regenerate as *const (),
        ffi::frost_clear_context as *const (),
        ffi::frost_cancel as *const (),
        ffi::frost_set_mode as *const (),
        ffi::frost_set_repo as *const (),
        ffi::frost_last_error as *const (),
        ffi::frost_string_free as *const (),
    ];
    std::hint::black_box(keep);

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--headless-check") {
        std::process::exit(headless_check());
    }

    let engine = ffi::frost_engine_new(std::ptr::null());
    let cargs: Vec<CString> = args.iter().map(|a| CString::new(a.as_str()).unwrap_or_default()).collect();
    let argv: Vec<*const c_char> = cargs.iter().map(|c| c.as_ptr()).collect();
    let code = unsafe { frost_ui_run(engine, argv.len() as c_int, argv.as_ptr()) };
    ffi::frost_engine_free(engine);
    std::process::exit(code);
}

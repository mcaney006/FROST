//! FROST desktop shell: a system-webview window (wry) whose backend is the SAME
//! frost-service::Engine used by the CLI. Inference, policy and persistence stay
//! in Rust; the webview is a thin front end that talks over wry IPC.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use frost_core::{Candidate, Decision, Mode, Request, ThermalState};
use frost_model::Embedder;
use frost_service::Engine;
use serde::Deserialize;
use std::rc::Rc;
use tao::{
    event::{Event, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};
use wry::WebViewBuilder;

const UI: &str = include_str!("../ui/index.html");

#[derive(Deserialize)]
struct Msg {
    cmd: String,
    #[serde(default)] state: String,
    #[serde(default)] actions: Vec<Act>,
    #[serde(default)] mode: String,
    #[serde(default)] thermal: String,
}
#[derive(Deserialize)]
struct Act { #[serde(default)] id: String, #[serde(default)] text: String }

enum UserEvent { Ipc(String) }

#[derive(Deserialize)]
struct IpcEnvelope {
    #[serde(default)] cmd: String,
    #[serde(default)] req: Option<u64>,
}

fn ipc_envelope(raw: &str) -> Option<IpcEnvelope> {
    serde_json::from_str(raw).ok()
}

fn parse_mode(s: &str) -> Mode {
    match s { "balanced" => Mode::Balanced, "performance" => Mode::Performance, _ => Mode::Quiet }
}
fn parse_thermal(s: &str) -> Option<ThermalState> {
    match s {
        "nominal" => Some(ThermalState::Nominal),
        "fair" => Some(ThermalState::Fair),
        "serious" => Some(ThermalState::Serious),
        "critical" => Some(ThermalState::Critical),
        _ => None,
    }
}

fn handle(engine: &Engine, raw: &str) -> String {
    let msg: Msg = match serde_json::from_str(raw) {
        Ok(m) => m,
        Err(e) => return format!(r#"{{"error":"bad request: {e}"}}"#),
    };
    match msg.cmd.as_str() {
        "info" => serde_json::json!({
            "fingerprint": engine.fingerprint(),
            "revision": frost_model::REVISION,
            "scorer": engine.scorer(),
            "device": "gpu",
        }).to_string(),
        "rank" => {
            engine.set_thermal_override(parse_thermal(&msg.thermal));
            let candidates: Vec<Candidate> = msg.actions.into_iter().enumerate()
                .filter(|(_, a)| !a.text.trim().is_empty())
                .map(|(i, a)| Candidate {
                    id: if a.id.trim().is_empty() { format!("a{i}") } else { a.id },
                    text: a.text, permission: None,
                }).collect();
            let req = Request { state: msg.state, candidates, mode: parse_mode(&msg.mode), allowed: None };
            let d: Decision = engine.rank(&req);
            serde_json::to_string(&d).unwrap_or_else(|e| format!(r#"{{"error":"{e}"}}"#))
        }
        "shutdown" => r#"{"bye":true}"#.to_string(),
        other => format!(r#"{{"error":"unknown cmd {other}"}}"#),
    }
}

fn main() -> wry::Result<()> {
    // Headless backend self-test: exercises the SAME handle()+Engine path the
    // webview uses, without opening a window. `frost-desktop --selftest`.
    if std::env::args().any(|a| a == "--selftest") {
        let dir = Embedder::default_dir();
        let engine = Engine::load(&dir).expect("engine load");
        let info = handle(&engine, r#"{"cmd":"info"}"#);
        println!("info => {info}");
        let rank = handle(&engine, r#"{"cmd":"rank","state":"play some music","mode":"quiet","actions":[{"id":"play","text":"play a song"},{"id":"delete","text":"erase all files"}]}"#);
        println!("rank => {rank}");
        assert!(rank.contains("\"kind\":\"ranked\"") && rank.contains("play"), "selftest rank failed");
        println!("SELFTEST OK");
        return Ok(());
    }
    let dir = Embedder::default_dir();
    let engine = match Engine::load(&dir) {
        Ok(e) => Rc::new(e),
        Err(e) => {
            // Show a helpful page instead of crashing when weights are missing.
            let html = format!("<h2 style='font-family:sans-serif'>FROST</h2><p style='font-family:sans-serif'>Model not loaded: {e}.<br>Run <code>bootstrap.sh</code> to download the weights, then relaunch.</p>");
            let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
            let window = WindowBuilder::new().with_title("FROST").build(&event_loop).unwrap();
            let _wv = WebViewBuilder::new().with_html(html).build(&window)?;
            event_loop.run(move |ev, _, cf| {
                *cf = ControlFlow::Wait;
                if let Event::WindowEvent { event: WindowEvent::CloseRequested, .. } = ev { *cf = ControlFlow::Exit; }
            });
        }
    };

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    let window = WindowBuilder::new()
        .with_title("FROST")
        .with_inner_size(tao::dpi::LogicalSize::new(960.0, 720.0))
        .build(&event_loop)
        .unwrap();

    let webview = Rc::new(
        WebViewBuilder::new()
            .with_html(UI)
            .with_ipc_handler(move |req| { let _ = proxy.send_event(UserEvent::Ipc(req.into_body())); })
            .build(&window)?,
    );

    let wv = webview.clone();
    let eng = engine.clone();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::NewEvents(StartCause::Init) => {}
            Event::UserEvent(UserEvent::Ipc(body)) => {
                let envelope = ipc_envelope(&body);
                if matches!(envelope.as_ref().map(|msg| msg.cmd.as_str()), Some("shutdown")) { *control_flow = ControlFlow::Exit; return; }
                let req = envelope.and_then(|msg| msg.req);
                let resp = handle(&eng, &body);
                let js = match req {
                    Some(req) => format!("window.__frost_resp({req},{resp})"),
                    None => format!("window.__frost_resp(null,{resp})"),
                };
                let _ = wv.evaluate_script(&js);
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

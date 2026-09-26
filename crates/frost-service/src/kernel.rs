//! frost-service: owns long-lived native workers and the shared engine plumbing.
//!
//! `MojoKernel` is a persistent coprocessor: ONE compiled-Mojo helper process,
//! never spawned per request, driven over a bounded line-framed pipe protocol.
//! This is the contract-sanctioned route for Mojo numerical work on this
//! toolchain nightly (direct buffer C-ABI export is blocked upstream).

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("helper binary not found at {0}")]
    NotFound(PathBuf),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("helper closed the pipe")]
    Closed,
    #[error("bad response: {0}")]
    BadResponse(String),
}

/// Persistent Mojo kernel coprocessor. Single-flight (Mutex) — the initial
/// conservative config runs one inference request at a time.
pub struct MojoKernel {
    inner: Mutex<Proc>,
    mem_rss_kb_at_start: u64,
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl MojoKernel {
    /// Default helper location inside the source tree / install root.
    pub fn default_path() -> PathBuf {
        // Resolution order: explicit env, installed app-support bin, next to the
        // running executable, then the dev source tree. Lets the SAME code find
        // the compiled Mojo helper whether running installed or from `cargo`.
        if let Ok(p) = std::env::var("FROST_MOJO_HELPER") {
            let pb = PathBuf::from(p);
            if pb.exists() { return pb; }
        }
        if let Ok(home) = std::env::var("HOME") {
            let pb = PathBuf::from(home).join("Library/Application Support/FROST/bin/frost_mojo_helper");
            if pb.exists() { return pb; }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                for cand in [dir.join("frost_mojo_helper"), dir.join("../Resources/frost_mojo_helper")] {
                    if cand.exists() { return cand; }
                }
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap()
            .join("native/kernels/frost_mojo_helper")
    }

    pub fn spawn(path: &PathBuf) -> Result<Self, KernelError> {
        if !path.exists() {
            return Err(KernelError::NotFound(path.clone()));
        }
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().ok_or(KernelError::Closed)?;
        let stdout = BufReader::new(child.stdout.take().ok_or(KernelError::Closed)?);
        let pid = child.id();
        Ok(Self {
            inner: Mutex::new(Proc { child, stdin, stdout }),
            mem_rss_kb_at_start: rss_kb(pid).unwrap_or(0),
        })
    }

    fn request(&self, line: &str) -> Result<String, KernelError> {
        let mut g = self.inner.lock().unwrap();
        g.stdin.write_all(line.as_bytes())?;
        g.stdin.write_all(b"\n")?;
        g.stdin.flush()?;
        let mut resp = String::new();
        let n = g.stdout.read_line(&mut resp)?;
        if n == 0 {
            return Err(KernelError::Closed);
        }
        Ok(resp.trim_end().to_string())
    }

    /// Liveness check.
    pub fn ping(&self) -> Result<bool, KernelError> {
        Ok(self.request("PING")? == "PONG")
    }

    /// L2-normalize a vector in compiled Mojo.
    pub fn normalize(&self, v: &[f32]) -> Result<Vec<f32>, KernelError> {
        let mut req = format!("NORM {}", v.len());
        for x in v {
            req.push(' ');
            req.push_str(&fmt_f32(*x));
        }
        parse_floats(&self.request(&req)?)
    }

    /// Cosine rerank scores of query vs each candidate, computed in Mojo.
    pub fn score(&self, q: &[f32], cands: &[Vec<f32>]) -> Result<Vec<f32>, KernelError> {
        let d = q.len();
        let mut req = format!("SCORE {} {}", d, cands.len());
        for x in q {
            req.push(' ');
            req.push_str(&fmt_f32(*x));
        }
        for c in cands {
            assert_eq!(c.len(), d, "candidate dim must match query dim");
            for x in c {
                req.push(' ');
                req.push_str(&fmt_f32(*x));
            }
        }
        parse_floats(&self.request(&req)?)
    }

    /// Approximate resident-set overhead of the helper (KiB), for the record.
    pub fn mem_overhead_kb(&self) -> u64 {
        let pid = { self.inner.lock().unwrap().child.id() };
        rss_kb(pid).unwrap_or(self.mem_rss_kb_at_start)
    }
}

impl Drop for MojoKernel {
    fn drop(&mut self) {
        if let Ok(mut g) = self.inner.lock() {
            // Empty line terminates the helper loop; then reap.
            let _ = g.stdin.write_all(b"\n");
            let _ = g.stdin.flush();
            let _ = g.child.wait();
        }
    }
}

fn fmt_f32(x: f32) -> String {
    // Compact, round-trippable enough for the kernel; avoids locale issues.
    format!("{x:.7}")
}

fn parse_floats(s: &str) -> Result<Vec<f32>, KernelError> {
    if s.starts_with("ERR") {
        return Err(KernelError::BadResponse(s.to_string()));
    }
    s.split_whitespace()
        .map(|t| t.parse::<f32>().map_err(|e| KernelError::BadResponse(format!("{t}: {e}"))))
        .collect()
}

/// RSS in KiB for a pid via `ps` (best-effort; used only for reporting).
fn rss_kb(pid: u32) -> Option<u64> {
    let out = Command::new("ps").args(["-o", "rss=", "-p", &pid.to_string()]).output().ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mojo_kernel_roundtrip() {
        let path = MojoKernel::default_path();
        if !path.exists() {
            eprintln!("SKIP: helper not built at {}", path.display());
            return;
        }
        let k = MojoKernel::spawn(&path).expect("spawn helper");
        assert!(k.ping().expect("ping"));
        let n = k.normalize(&[3.0, 4.0]).expect("norm");
        assert!((n[0] - 0.6).abs() < 1e-4 && (n[1] - 0.8).abs() < 1e-4, "{n:?}");
        let s = k.score(&[1.0, 0.0], &[vec![1.0, 0.0], vec![0.0, 1.0]]).expect("score");
        assert!((s[0] - 1.0).abs() < 1e-4 && s[1].abs() < 1e-4, "{s:?}");
        eprintln!("mojo helper RSS ~= {} KiB", k.mem_overhead_kb());
    }
}

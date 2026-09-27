//! Safety-critical execution layer for the FROST coding assistant.
//!
//! Three pieces, none of which decides to do anything on its own — the caller
//! (an approval gate) decides, this crate makes the decided thing bounded:
//!
//! * [`Workspace`] — every path argument goes through [`Workspace::resolve`],
//!   which rejects absolute paths, `..`, symlinks leaving the root, and
//!   dependency/VCS/secret-like names. The read-only tools ([`Workspace::list`],
//!   [`Workspace::read`], [`Workspace::search`], [`Workspace::diagnostics`])
//!   never touch anything outside the root; `diagnostics` only *plans* a command.
//! * [`patch`] — strict unified-diff parsing, base-hash-pinned validation with
//!   bounded (±3 line, unique-match) fuzz, atomic all-or-nothing apply, revert.
//! * [`run`] — a bounded process runner: scrubbed env, no stdin, no shell, own
//!   process group, capped output, hard deadline, cooperative cancel.
//!
//! # This is NOT a sandbox
//!
//! [`run::run`] executes real programs with the user's full privileges:
//!
//! * `cargo test`, `cargo build`, `zig build` and friends run build scripts,
//!   proc macros and test binaries — arbitrary code from the workspace.
//! * The working directory is where the process *starts*, not confinement; the
//!   child can read and write anything the user can.
//! * Network access is not blocked.
//! * Environment scrubbing and process-group kill bound *leakage* and
//!   *lifetime*, not capability.
//!
//! Treat every approved command as "the user ran this in a terminal".

pub mod patch;
pub mod run;

pub use patch::{apply, parse_unified_diff, revert, validate, ApplyReport, Patch, Validated};
pub use run::{run, RunResult, RunSpec, DEFAULT_ENV_ALLOWLIST};

use serde::Serialize;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

/// Files larger than this are refused by [`Workspace::read`] and the patch engine.
pub const MAX_TEXT_FILE_BYTES: u64 = 8 * 1024 * 1024;
/// Files larger than this are skipped by [`Workspace::search`].
pub const MAX_SEARCH_FILE_BYTES: u64 = 512 * 1024;
/// Hit lines longer than this are cut (at a char boundary) in search results.
const MAX_HIT_TEXT_BYTES: usize = 400;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolError {
    #[error("absolute path not allowed: {path}")]
    AbsolutePath { path: String },
    #[error("path escapes the workspace: {path}")]
    Escape { path: String },
    #[error("path is excluded (dependency, VCS, or secret-like name): {path}")]
    Excluded { path: String },
    #[error("not found: {path}")]
    NotFound { path: String },
    #[error("{path}: {msg}")]
    Io { path: String, msg: String },
    #[error("not a text file (binary, non-UTF-8, or larger than 8 MiB): {path}")]
    NotText { path: String },
    #[error("invalid argument: {msg}")]
    InvalidArgument { msg: String },
    #[error("diff parse error at line {line}: {msg}")]
    Parse { line: usize, msg: String },
    #[error("unsupported diff feature: {what}")]
    Unsupported { what: String },
    #[error("hunk {hunk} of {path} does not apply: {msg}")]
    HunkMismatch { path: String, hunk: usize, msg: String },
    #[error("stale patch for {path}: expected base {expected}, found {actual}")]
    StalePatch { path: String, expected: String, actual: String },
    #[error("no expected base hash supplied for {path}")]
    MissingBaseHash { path: String },
    #[error("apply failed on {path}: {msg} (rolled back: {rolled_back})")]
    ApplyFailed { path: String, msg: String, rolled_back: bool },
    #[error("command rejected: {msg}")]
    BadCommand { msg: String },
    #[error("failed to spawn {program}: {msg}")]
    Spawn { program: String, msg: String },
}

pub(crate) fn io_err(path: &str, e: std::io::Error) -> ToolError {
    if e.kind() == std::io::ErrorKind::NotFound {
        ToolError::NotFound { path: path.to_string() }
    } else {
        ToolError::Io { path: path.to_string(), msg: e.to_string() }
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Same exclusion set as the indexer: dependency/build dirs, VCS metadata, and
/// secret-like file names. Checked per path component, case-insensitively.
pub fn is_excluded_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    matches!(n.as_str(), ".git" | "target" | "node_modules")
        || n.starts_with(".env")
        || n.starts_with("id_rsa")
        || n.ends_with(".pem")
        || n.ends_with(".key")
        || n.ends_with(".sqlite")
        || n.contains("credentials")
        || n.contains("secret")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticsKind {
    Cargo,
    Zig,
    Clang,
}

/// A command the caller may choose to run (after approval) via [`run::run`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticsPlan {
    pub kind: DiagnosticsKind,
    /// Workspace-relative working directory.
    pub cwd: String,
    pub argv: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ListEntry {
    pub name: String,
    pub is_dir: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ListResult {
    pub path: String,
    pub entries: Vec<ListEntry>,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadResult {
    pub path: String,
    /// 1-based, inclusive. `end_line < start_line` means no lines returned.
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub total_lines: usize,
    /// SHA-256 of the whole file — the base hash to pin a later patch to.
    pub sha256_hex: String,
    /// `max_bytes` cut the requested range short.
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchHit {
    pub path: String,
    pub line: usize,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    /// More hits existed beyond `max_hits`.
    pub truncated: bool,
    pub files_scanned: usize,
    /// Files skipped as too large or binary.
    pub files_skipped: usize,
}

#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
    allow_hidden: bool,
}

impl Workspace {
    /// Canonicalizes `root`; it must be an existing directory.
    pub fn open(root: impl AsRef<Path>) -> Result<Workspace, ToolError> {
        let shown = root.as_ref().display().to_string();
        let root = fs::canonicalize(root.as_ref()).map_err(|e| io_err(&shown, e))?;
        if !root.is_dir() {
            return Err(ToolError::InvalidArgument { msg: format!("workspace root is not a directory: {shown}") });
        }
        Ok(Workspace { root, allow_hidden: false })
    }

    /// Lifts the dependency/VCS/secret exclusions. Escape checks always apply.
    pub fn with_allow_hidden(mut self, allow: bool) -> Workspace {
        self.allow_hidden = allow;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves a workspace-relative path to an absolute one inside the root.
    /// The target need not exist (writes), but every existing ancestor is
    /// canonicalized, so symlinks cannot lead outside the root or onto an
    /// excluded name.
    pub fn resolve(&self, rel: &str) -> Result<PathBuf, ToolError> {
        self.resolve_with_rel(rel).map(|(abs, _)| abs)
    }

    /// `(absolute path, normalized relative path with '/' separators)`.
    pub(crate) fn resolve_with_rel(&self, rel: &str) -> Result<(PathBuf, String), ToolError> {
        let shown = || rel.to_string();
        if rel.contains('\0') {
            return Err(ToolError::InvalidArgument { msg: "path contains NUL".into() });
        }
        let mut names: Vec<&str> = Vec::new();
        for c in Path::new(rel).components() {
            match c {
                Component::CurDir => {}
                Component::Normal(n) => {
                    let n = n.to_str().ok_or_else(|| ToolError::InvalidArgument { msg: "path is not UTF-8".into() })?;
                    names.push(n);
                }
                Component::ParentDir => return Err(ToolError::Escape { path: shown() }),
                Component::RootDir | Component::Prefix(_) => return Err(ToolError::AbsolutePath { path: shown() }),
            }
        }
        if !self.allow_hidden && names.iter().any(|n| is_excluded_name(n)) {
            return Err(ToolError::Excluded { path: shown() });
        }

        // Canonicalize the deepest ancestor that exists (as a file, dir, or
        // symlink). A symlink that cannot be canonicalized (dangling, loop) is
        // refused: writing through it could create a file anywhere.
        let mut split = names.len();
        let base = loop {
            let candidate = names[..split].iter().fold(self.root.clone(), |p, n| p.join(n));
            if fs::symlink_metadata(&candidate).is_ok() {
                break fs::canonicalize(&candidate).map_err(|_| ToolError::Escape { path: shown() })?;
            }
            // `split == 0` is the root itself, which exists.
            split -= 1;
        };
        let inside = base.strip_prefix(&self.root).map_err(|_| ToolError::Escape { path: shown() })?;
        let mut rel_parts: Vec<String> = Vec::new();
        for c in inside.components() {
            let n = c.as_os_str().to_string_lossy().into_owned();
            if !self.allow_hidden && is_excluded_name(&n) {
                return Err(ToolError::Excluded { path: shown() });
            }
            rel_parts.push(n);
        }
        let mut abs = base;
        for n in &names[split..] {
            abs.push(n);
            rel_parts.push((*n).to_string());
        }
        Ok((abs, rel_parts.join("/")))
    }

    /// Sorted directory listing (directories marked), excluded names skipped.
    pub fn list(&self, rel_dir: &str, max_entries: usize) -> Result<ListResult, ToolError> {
        let (abs, rel) = self.resolve_with_rel(rel_dir)?;
        let rd = fs::read_dir(&abs).map_err(|e| io_err(&rel, e))?;
        let mut entries = Vec::new();
        for e in rd {
            let e = e.map_err(|e| io_err(&rel, e))?;
            let Ok(name) = e.file_name().into_string() else { continue };
            if !self.allow_hidden && is_excluded_name(&name) {
                continue;
            }
            // file_type() does not follow symlinks: a link is never shown as a dir.
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            entries.push(ListEntry { name, is_dir });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let truncated = entries.len() > max_entries;
        entries.truncate(max_entries);
        Ok(ListResult { path: rel, entries, truncated })
    }

    /// Reads lines `start_line..=end_line` (1-based; `0` start means 1, `0`
    /// end means EOF), capping returned content at `max_bytes`.
    pub fn read(&self, rel_path: &str, start_line: usize, end_line: usize, max_bytes: usize) -> Result<ReadResult, ToolError> {
        let (abs, rel) = self.resolve_with_rel(rel_path)?;
        let bytes = read_text_file(&abs, &rel)?;
        let text = std::str::from_utf8(&bytes).map_err(|_| ToolError::NotText { path: rel.clone() })?;
        let lines: Vec<&str> = text.split_inclusive('\n').collect();
        let total_lines = lines.len();
        let start = start_line.max(1);
        let end = if end_line == 0 { total_lines } else { end_line.min(total_lines) };
        let mut content = String::new();
        let mut last = start.saturating_sub(1);
        let mut truncated = false;
        for (i, line) in lines.iter().enumerate().take(end).skip(start - 1) {
            if content.len() + line.len() > max_bytes {
                truncated = true;
                if content.is_empty() {
                    // Not even one full line fits: return its prefix.
                    content.push_str(cut_at_char_boundary(line, max_bytes));
                    last = i + 1;
                }
                break;
            }
            content.push_str(line);
            last = i + 1;
        }
        Ok(ReadResult { path: rel, start_line: start, end_line: last, content, total_lines, sha256_hex: sha256_hex(&bytes), truncated })
    }

    /// Literal substring search over every non-excluded text file (no regex,
    /// symlinks not followed, files > 512 KiB or binary skipped).
    pub fn search(&self, pattern_literal: &str, max_hits: usize, case_insensitive: bool) -> Result<SearchResult, ToolError> {
        if pattern_literal.is_empty() {
            return Err(ToolError::InvalidArgument { msg: "empty search pattern".into() });
        }
        let needle = if case_insensitive { pattern_literal.to_lowercase() } else { pattern_literal.to_string() };
        let mut out = SearchResult { hits: Vec::new(), truncated: false, files_scanned: 0, files_skipped: 0 };
        self.search_dir(&self.root.clone(), &needle, case_insensitive, max_hits, &mut out)?;
        Ok(out)
    }

    fn search_dir(&self, dir: &Path, needle: &str, ci: bool, max_hits: usize, out: &mut SearchResult) -> Result<(), ToolError> {
        let rel_of = |p: &Path| p.strip_prefix(&self.root).unwrap_or(p).to_string_lossy().replace('\\', "/");
        let mut entries: Vec<_> = fs::read_dir(dir).map_err(|e| io_err(&rel_of(dir), e))?.filter_map(Result::ok).collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            if out.truncated {
                return Ok(());
            }
            let Ok(name) = e.file_name().into_string() else { continue };
            if !self.allow_hidden && is_excluded_name(&name) {
                continue;
            }
            let Ok(ft) = e.file_type() else { continue };
            let path = e.path();
            if ft.is_dir() {
                self.search_dir(&path, needle, ci, max_hits, out)?;
                continue;
            }
            if !ft.is_file() {
                continue; // symlinks and specials are never followed
            }
            let too_big = e.metadata().map(|m| m.len() > MAX_SEARCH_FILE_BYTES).unwrap_or(true);
            let Some(text) = (!too_big).then(|| fs::read(&path).ok()).flatten().filter(|b| !b.contains(&0)) else {
                out.files_skipped += 1;
                continue;
            };
            out.files_scanned += 1;
            let text = String::from_utf8_lossy(&text);
            for (i, line) in text.lines().enumerate() {
                let found = if ci { line.to_lowercase().contains(needle) } else { line.contains(needle) };
                if !found {
                    continue;
                }
                if out.hits.len() == max_hits {
                    out.truncated = true;
                    return Ok(());
                }
                out.hits.push(SearchHit { path: rel_of(&path), line: i + 1, text: cut_at_char_boundary(line, MAX_HIT_TEXT_BYTES).to_string() });
            }
        }
        Ok(())
    }

    /// Plans (does NOT run) the diagnostics command for `rel_dir`.
    pub fn diagnostics(&self, kind: DiagnosticsKind, rel_dir: &str) -> Result<DiagnosticsPlan, ToolError> {
        let (abs, rel) = self.resolve_with_rel(rel_dir)?;
        if !abs.is_dir() {
            return Err(ToolError::InvalidArgument { msg: format!("not a directory: {rel_dir}") });
        }
        let cwd = if rel.is_empty() { ".".to_string() } else { rel };
        let files_with = |ext: &str| -> Result<Vec<String>, ToolError> {
            let mut v: Vec<String> = self.list(&cwd, usize::MAX)?.entries.into_iter().filter(|e| !e.is_dir && e.name.ends_with(ext)).map(|e| e.name).collect();
            v.sort();
            Ok(v)
        };
        let argv: Vec<String> = match kind {
            DiagnosticsKind::Cargo => ["cargo", "check", "--message-format", "short"].map(String::from).to_vec(),
            DiagnosticsKind::Zig if abs.join("build.zig").is_file() => ["zig", "build", "test"].map(String::from).to_vec(),
            DiagnosticsKind::Zig => {
                let zig = files_with(".zig")?;
                // Prefer a test entry point; otherwise the first file.
                let file = zig.iter().find(|f| f.ends_with("test.zig")).or(zig.first()).ok_or_else(|| ToolError::InvalidArgument { msg: format!("no build.zig or .zig files in {cwd}") })?;
                // Workspace-local cache: zig 0.16's default (global) cache for a
                // standalone `zig test` can serve a stale result to a copy of
                // the same files at another path (observed on the fixtures).
                ["zig", "test", "--cache-dir", ".zig-cache"].map(String::from).into_iter().chain([file.clone()]).collect()
            }
            DiagnosticsKind::Clang => {
                let c = files_with(".c")?;
                if c.is_empty() {
                    return Err(ToolError::InvalidArgument { msg: format!("no .c files in {cwd}") });
                }
                ["clang", "-fsyntax-only"].map(String::from).into_iter().chain(c).collect()
            }
        };
        Ok(DiagnosticsPlan { kind, cwd, argv })
    }
}

/// Reads a regular file capped at [`MAX_TEXT_FILE_BYTES`], refusing binary content.
pub(crate) fn read_text_file(abs: &Path, rel: &str) -> Result<Vec<u8>, ToolError> {
    let f = fs::File::open(abs).map_err(|e| io_err(rel, e))?;
    let mut bytes = Vec::new();
    f.take(MAX_TEXT_FILE_BYTES + 1).read_to_end(&mut bytes).map_err(|e| io_err(rel, e))?;
    if bytes.len() as u64 > MAX_TEXT_FILE_BYTES || bytes.contains(&0) {
        return Err(ToolError::NotText { path: rel.to_string() });
    }
    Ok(bytes)
}

fn cut_at_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

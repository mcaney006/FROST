//! Unified-diff patch engine: strict parse → base-hash-pinned validate →
//! atomic all-or-nothing apply → revert.
//!
//! Supported: multi-file diffs (`--- a/x` / `+++ b/x`, prefixes optional),
//! `@@ -l[,c] +l[,c] @@[ trailer]` hunks, `\ No newline at end of file`,
//! creations (`--- /dev/null`) and deletions (`+++ /dev/null`). Rejected:
//! renames, copies, binary patches, quoted paths.
//!
//! Tolerant where models are sloppy, strict where it matters: hunk-header
//! *counts* are advisory (a hunk ends at the next `@@`, file header, non-hunk
//! line, or EOF, and counts are recomputed from the body) and a wrong *start
//! line* is recovered by locating the exact pre-image (context + removed
//! lines) at a unique position in the file. The pre-image itself must still
//! match byte-for-byte, and base hashes and workspace confinement are unchanged.

use crate::{io_err, read_text_file, sha256_hex, ToolError, Workspace};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Base-hash token for a file the patch creates (it must not exist yet).
pub const NEW_FILE: &str = "new";
/// After-hash token for a file the patch deletes.
pub const DELETED_FILE: &str = "deleted";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    pub files: Vec<FilePatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePatch {
    /// `None` = file is created.
    pub old_path: Option<String>,
    /// `None` = file is deleted.
    pub new_path: Option<String>,
    pub hunks: Vec<Hunk>,
}

impl FilePatch {
    pub fn path(&self) -> &str {
        self.new_path.as_deref().or(self.old_path.as_deref()).unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hunk {
    /// As stated in the header (may be wrong; see [`validate`]).
    pub old_start: usize,
    /// Recomputed from the body; header counts are ignored.
    pub old_count: usize,
    pub new_start: usize,
    pub new_count: usize,
    pub lines: Vec<HunkLine>,
    /// The old side's last line has no trailing newline.
    pub old_no_newline: bool,
    /// The new side's last line has no trailing newline.
    pub new_no_newline: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HunkLine {
    Context(String),
    Remove(String),
    Add(String),
}

impl Patch {
    pub fn affected_files(&self) -> Vec<String> {
        self.files.iter().map(|f| f.path().to_string()).collect()
    }

    /// SHA-256 of each current pre-image file, or [`NEW_FILE`] for creations.
    /// Pin these at proposal time and pass them to [`validate`].
    pub fn base_hashes(&self, ws: &Workspace) -> Result<BTreeMap<String, String>, ToolError> {
        let mut out = BTreeMap::new();
        for f in &self.files {
            let hash = match f.old_path {
                None => NEW_FILE.to_string(),
                Some(_) => {
                    let (abs, rel) = ws.resolve_with_rel(f.path())?;
                    sha256_hex(&read_text_file(&abs, &rel)?)
                }
            };
            out.insert(f.path().to_string(), hash);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- parsing

fn parse_err(line: usize, msg: impl Into<String>) -> ToolError {
    ToolError::Parse { line, msg: msg.into() }
}

fn header_path(raw: &str, prefix: &str, line: usize) -> Result<Option<String>, ToolError> {
    // Strip an optional "\t<timestamp>" suffix.
    let p = raw.split('\t').next().unwrap_or_default().trim_end();
    if p.starts_with('"') {
        return Err(ToolError::Unsupported { what: format!("quoted path {p}") });
    }
    if p == "/dev/null" {
        return Ok(None);
    }
    let p = p.strip_prefix(prefix).unwrap_or(p);
    if p.is_empty() {
        return Err(parse_err(line, "empty path in file header"));
    }
    Ok(Some(p.to_string()))
}

/// Start line of a `-l[,c]` / `+l[,c]` range. The count is advisory and ignored.
fn range_start(s: Option<&str>, sign: char, line: usize) -> Result<usize, ToolError> {
    let s = s.and_then(|s| s.strip_prefix(sign)).ok_or_else(|| parse_err(line, format!("expected '{sign}' range in hunk header")))?;
    let start = s.split_once(',').map_or(s, |(a, _)| a);
    start.parse::<usize>().map_err(|_| parse_err(line, format!("bad start line '{start}' in hunk header")))
}

/// `--- x` / `+++ y` / `@@` in a row: a new file section, even mid-hunk (the
/// `@@` requirement keeps a removed "-- x" line followed by an added "++ y"
/// line from being mistaken for one).
fn is_file_header(lines: &[&str], j: usize) -> bool {
    lines[j].starts_with("--- ") && lines.get(j + 1).is_some_and(|l| l.starts_with("+++ ")) && lines.get(j + 2).is_some_and(|l| l.starts_with("@@"))
}

/// Parses one hunk starting at `lines[i]` (the `@@` line). Header counts are
/// ignored: the body runs until the next `@@`, file header, non-hunk line, or
/// EOF, and the counts are recomputed from it. Returns the hunk and the index
/// of the first line after it.
fn parse_hunk(lines: &[&str], i: usize) -> Result<(Hunk, usize), ToolError> {
    let rest = lines[i].trim_end_matches('\r').strip_prefix("@@").unwrap_or_default();
    // Anything after the closing "@@" is a trailer (function name etc.).
    let (ranges, _trailer) = rest.split_once("@@").ok_or_else(|| parse_err(i + 1, "hunk header has no closing '@@'"))?;
    let mut r = ranges.split_whitespace();
    let old_start = range_start(r.next(), '-', i + 1)?;
    let new_start = range_start(r.next(), '+', i + 1)?;
    if r.next().is_some() {
        return Err(parse_err(i + 1, "unexpected text between '@@' markers"));
    }
    let mut h = Hunk { old_start, old_count: 0, new_start, new_count: 0, lines: Vec::new(), old_no_newline: false, new_no_newline: false };
    let mut trailing_blank = 0; // raw-empty lines at the end are formatting, not context
    let mut j = i + 1;
    while j < lines.len() && !lines[j].starts_with("@@") && !is_file_header(lines, j) {
        let raw = lines[j];
        let line = match raw.as_bytes().first() {
            Some(b' ') => HunkLine::Context(raw[1..].to_string()),
            Some(b'-') => HunkLine::Remove(raw[1..].to_string()),
            Some(b'+') => HunkLine::Add(raw[1..].to_string()),
            // Some tools strip the lone space of an empty context line.
            None => HunkLine::Context(String::new()),
            Some(b'\\') => {
                // "\ No newline at end of file" (text varies by locale).
                match h.lines.last() {
                    Some(HunkLine::Context(_)) => (h.old_no_newline, h.new_no_newline) = (true, true),
                    Some(HunkLine::Remove(_)) => h.old_no_newline = true,
                    Some(HunkLine::Add(_)) => h.new_no_newline = true,
                    None => return Err(parse_err(j + 1, "no-newline marker before any hunk line")),
                }
                trailing_blank = 0;
                j += 1;
                continue;
            }
            Some(_) => break, // not a hunk line: the caller decides whether it is a legal header
        };
        trailing_blank = if raw.is_empty() { trailing_blank + 1 } else { 0 };
        h.lines.push(line);
        j += 1;
    }
    h.lines.truncate(h.lines.len() - trailing_blank);
    if h.lines.is_empty() {
        return Err(parse_err(i + 1, "empty hunk"));
    }
    h.old_count = h.lines.iter().filter(|l| !matches!(l, HunkLine::Add(_))).count();
    h.new_count = h.lines.iter().filter(|l| !matches!(l, HunkLine::Remove(_))).count();
    Ok((h, j))
}

const IGNORED_HEADERS: &[&str] = &["diff ", "index ", "new file mode", "deleted file mode", "old mode", "new mode"];
const UNSUPPORTED_HEADERS: &[&str] = &["rename from", "rename to", "copy from", "copy to", "similarity index", "dissimilarity index", "Binary files", "GIT binary patch"];

/// Parses a (possibly multi-file) unified diff. Strict: after the first file
/// header only known git header lines may appear between hunks.
pub fn parse_unified_diff(text: &str) -> Result<Patch, ToolError> {
    let mut lines: Vec<&str> = text.split('\n').collect();
    if text.ends_with('\n') {
        lines.pop();
    }
    let mut files: Vec<FilePatch> = Vec::new();
    let mut seen = BTreeSet::new();
    let mut i = 0;
    while i < lines.len() {
        let hdr = lines[i].trim_end_matches('\r');
        if let Some(what) = UNSUPPORTED_HEADERS.iter().find(|p| hdr.starts_with(*p)) {
            return Err(ToolError::Unsupported { what: format!("{what} (line {})", i + 1) });
        }
        if let Some(old_raw) = hdr.strip_prefix("--- ") {
            let new_raw = lines
                .get(i + 1)
                .map(|l| l.trim_end_matches('\r'))
                .and_then(|l| l.strip_prefix("+++ "))
                .ok_or_else(|| parse_err(i + 2, "expected '+++' after '---'"))?;
            let old_path = header_path(old_raw, "a/", i + 1)?;
            let new_path = header_path(new_raw, "b/", i + 2)?;
            match (&old_path, &new_path) {
                (None, None) => return Err(parse_err(i + 1, "both sides are /dev/null")),
                (Some(a), Some(b)) if a != b => return Err(ToolError::Unsupported { what: format!("rename {a} -> {b}") }),
                _ => {}
            }
            i += 2;
            let mut hunks = Vec::new();
            while i < lines.len() && lines[i].starts_with("@@") {
                let (h, next) = parse_hunk(&lines, i)?;
                hunks.push(h);
                i = next;
            }
            let fp = FilePatch { old_path, new_path, hunks };
            let path = fp.path().to_string();
            let single = fp.hunks.len() == 1;
            if fp.hunks.is_empty() {
                return Err(parse_err(i + 1, format!("no hunks for {path}")));
            }
            // Counts here are the recomputed ones: a creation is only '+' lines, a deletion only '-'.
            if fp.old_path.is_none() && !(single && fp.hunks[0].old_start == 0 && fp.hunks[0].old_count == 0) {
                return Err(parse_err(i, format!("creation of {path} must be one '-0' hunk of added lines")));
            }
            if fp.new_path.is_none() && !(single && fp.hunks[0].new_start == 0 && fp.hunks[0].new_count == 0) {
                return Err(parse_err(i, format!("deletion of {path} must be one '+0' hunk of removed lines")));
            }
            if !seen.insert(path.clone()) {
                return Err(parse_err(i, format!("{path} appears twice in one patch")));
            }
            files.push(fp);
            continue;
        }
        if hdr.starts_with("@@") {
            return Err(parse_err(i + 1, "hunk outside a file section"));
        }
        // Before the first file, tolerate any preamble (commit message etc.).
        // After it, only git headers: any other line inside a file section
        // is a malformed hunk line and could not be applied safely.
        if !files.is_empty() && !hdr.is_empty() && !IGNORED_HEADERS.iter().any(|p| hdr.starts_with(p)) {
            return Err(parse_err(i + 1, "not a hunk line (hunk lines start with ' ', '+', '-' or '\\')"));
        }
        i += 1;
    }
    if files.is_empty() {
        return Err(parse_err(1, "no file sections found"));
    }
    Ok(Patch { files })
}

// ------------------------------------------------------------- validation

/// A patch checked against the current workspace, holding the in-memory
/// pre- and post-images. Only [`apply`] consumes it.
#[derive(Debug, Serialize)]
pub struct Validated {
    pub files: Vec<ValidatedFile>,
}

#[derive(Debug, Serialize)]
pub struct ValidatedFile {
    pub path: String,
    pub before_hash: String,
    pub after_hash: String,
    pub added: usize,
    pub removed: usize,
    /// Per hunk: actual position minus stated position (0 = exact).
    pub hunk_offsets: Vec<isize>,
    #[serde(skip)]
    abs: PathBuf,
    #[serde(skip)]
    before: Option<Vec<u8>>,
    #[serde(skip)]
    after: Option<Vec<u8>>,
    #[serde(skip)]
    mode: Option<u32>,
}

fn split_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    text.strip_suffix('\n').unwrap_or(text).split('\n').collect()
}

/// Applies `fp`'s hunks to `orig` in memory. Returns the new text and per-hunk offsets.
fn apply_hunks(path: &str, orig: &str, fp: &FilePatch) -> Result<(String, Vec<isize>), ToolError> {
    let orig_eol = orig.is_empty() || orig.ends_with('\n');
    let lines = split_lines(orig);
    let mut out: Vec<&str> = Vec::with_capacity(lines.len());
    let mut new_eol = orig_eol;
    let mut cursor = 0usize;
    let mut offsets = Vec::with_capacity(fp.hunks.len());
    for (hi, h) in fp.hunks.iter().enumerate() {
        let mismatch = |msg: String| ToolError::HunkMismatch { path: path.to_string(), hunk: hi + 1, msg };
        let old: Vec<&str> = h.lines.iter().filter_map(|l| match l { HunkLine::Context(s) | HunkLine::Remove(s) => Some(s.as_str()), HunkLine::Add(_) => None }).collect();
        let new: Vec<&str> = h.lines.iter().filter_map(|l| match l { HunkLine::Context(s) | HunkLine::Add(s) => Some(s.as_str()), HunkLine::Remove(_) => None }).collect();
        // Pure insertions state the line they follow; others state their first line.
        let stated = if old.is_empty() { h.old_start } else { h.old_start.saturating_sub(1) };
        let text_at = |pos: usize| pos + old.len() <= lines.len() && lines[pos..pos + old.len()] == old[..];
        let fits = |pos: usize| -> bool {
            if pos < cursor || !text_at(pos) {
                return false;
            }
            let at_eof = pos + old.len() == lines.len();
            // A present old-side marker must be true; a missing one is tolerated.
            (!h.old_no_newline || (at_eof && !orig_eol && !lines.is_empty())) && (!h.new_no_newline || at_eof)
        };
        let pos = if fits(stated) {
            stated
        } else if old.is_empty() {
            return Err(mismatch(format!("insertion point after line {} is not usable", h.old_start)));
        } else {
            // Stated line is wrong: the exact pre-image must occur exactly once in the whole file.
            // ponytail: O(lines × pre-image) scan; fine for ≤8 MiB text files.
            let found: Vec<usize> = (0..=lines.len().saturating_sub(old.len())).filter(|&p| text_at(p)).collect();
            match found.as_slice() {
                [p] if fits(*p) => *p,
                [p] => return Err(mismatch(format!("pre-image found only at line {}, which overlaps an earlier hunk or contradicts its no-newline marker", p + 1))),
                [] => return Err(mismatch(format!("pre-image (context + removed lines) not found anywhere in the file; header said line {}", h.old_start))),
                many => {
                    let at: Vec<usize> = many.iter().map(|p| p + 1).collect();
                    return Err(mismatch(format!("pre-image is ambiguous: matches at lines {at:?}, and not at the stated line {}", h.old_start)));
                }
            }
        };
        let offset = pos as isize - stated as isize;
        out.extend_from_slice(&lines[cursor..pos]);
        out.extend_from_slice(&new);
        cursor = pos + old.len();
        if cursor == lines.len() {
            // This hunk owns the end of the file. With markers, follow them;
            // without any, keep the file's existing final-newline state.
            new_eol = if h.old_no_newline || h.new_no_newline { new.is_empty() || !h.new_no_newline } else { orig_eol || new.is_empty() };
        }
        offsets.push(offset);
    }
    out.extend_from_slice(&lines[cursor..]);
    let mut text = out.join("\n");
    if new_eol && !out.is_empty() {
        text.push('\n');
    }
    Ok((text, offsets))
}

/// Checks that every hunk's pre-image matches the current file byte-for-byte at
/// its stated line (or, if that line is wrong, at the one unique place it occurs
/// in the file; ambiguous or absent pre-images are rejected), that each file's current hash
/// equals the caller's pinned base hash, and that all paths resolve inside the
/// workspace. Nothing is written.
pub fn validate(patch: &Patch, ws: &Workspace, expected_base_hashes: &BTreeMap<String, String>) -> Result<Validated, ToolError> {
    let mut files = Vec::with_capacity(patch.files.len());
    for fp in &patch.files {
        let (abs, rel) = ws.resolve_with_rel(fp.path())?;
        let meta = fs::symlink_metadata(&abs).ok();
        let before = match &meta {
            Some(m) if !m.is_file() => return Err(ToolError::InvalidArgument { msg: format!("{rel} is not a regular file") }),
            Some(_) => Some(read_text_file(&abs, &rel)?),
            None => None,
        };
        let actual = before.as_deref().map_or_else(|| NEW_FILE.to_string(), sha256_hex);
        let expected = expected_base_hashes.get(fp.path()).ok_or_else(|| ToolError::MissingBaseHash { path: fp.path().to_string() })?;
        if *expected != actual {
            return Err(ToolError::StalePatch { path: rel, expected: expected.clone(), actual });
        }
        if fp.old_path.is_some() && before.is_none() {
            return Err(ToolError::NotFound { path: rel });
        }
        let orig = match &before {
            Some(b) => std::str::from_utf8(b).map_err(|_| ToolError::NotText { path: rel.clone() })?,
            None => "",
        };
        let (new_text, hunk_offsets) = apply_hunks(&rel, orig, fp)?;
        let after = match fp.new_path {
            Some(_) => Some(new_text.into_bytes()),
            None if new_text.is_empty() => None,
            None => return Err(ToolError::HunkMismatch { path: rel, hunk: 1, msg: "deletion does not remove the whole file".into() }),
        };
        let count = |f: fn(&HunkLine) -> bool| fp.hunks.iter().flat_map(|h| &h.lines).filter(|l| f(l)).count();
        files.push(ValidatedFile {
            path: rel,
            before_hash: actual,
            after_hash: after.as_deref().map_or_else(|| DELETED_FILE.to_string(), sha256_hex),
            added: count(|l| matches!(l, HunkLine::Add(_))),
            removed: count(|l| matches!(l, HunkLine::Remove(_))),
            hunk_offsets,
            abs,
            before,
            after,
            mode: meta.map(|m| m.permissions().mode()),
        });
    }
    Ok(Validated { files })
}

// ------------------------------------------------------------ apply/revert

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AppliedFile {
    pub path: String,
    pub before_hash: String,
    pub after_hash: String,
    pub added: usize,
    pub removed: usize,
}

/// What [`apply`] did, plus the in-memory pre-images [`revert`] needs.
#[derive(Debug, Serialize)]
pub struct ApplyReport {
    pub files: Vec<AppliedFile>,
    #[serde(skip)]
    undo: Vec<Undo>,
    #[serde(skip)]
    created_dirs: Vec<PathBuf>,
}

#[derive(Debug)]
struct Undo {
    path: String,
    abs: PathBuf,
    before: Option<Vec<u8>>,
    mode: Option<u32>,
    /// `None` when the patch deleted the file.
    after_hash: Option<String>,
}

fn current_hash(abs: &Path, rel: &str) -> Result<Option<String>, ToolError> {
    match fs::symlink_metadata(abs) {
        Err(_) => Ok(None),
        Ok(_) => Ok(Some(sha256_hex(&read_text_file(abs, rel)?))),
    }
}

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Writes via a temp file in the same directory + rename, so readers never see
/// a torn file. `create_new` refuses to follow anything already at the temp path.
fn write_atomic(abs: &Path, bytes: &[u8], mode: Option<u32>) -> std::io::Result<()> {
    let dir = abs.parent().ok_or_else(|| std::io::Error::other("path has no parent"))?;
    let name = abs.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.frost-tmp.{}.{}", std::process::id(), TMP_COUNTER.fetch_add(1, Ordering::Relaxed)));
    let result = (|| {
        let mut f = fs::OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(bytes)?;
        if let Some(m) = mode {
            f.set_permissions(fs::Permissions::from_mode(m))?;
        }
        f.sync_all()?;
        fs::rename(&tmp, abs)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Restores `before` (or removes the file if it did not exist). Returns success.
fn restore(abs: &Path, before: &Option<Vec<u8>>, mode: Option<u32>) -> bool {
    match before {
        Some(b) => write_atomic(abs, b, mode).is_ok(),
        None => fs::remove_file(abs).is_ok() || fs::symlink_metadata(abs).is_err(),
    }
}

/// Writes every file atomically. All-or-nothing: if any write fails, files
/// already written are restored from the in-memory pre-images and directories
/// created for new files are removed. Re-checks each base hash immediately
/// before writing, so a file edited after [`validate`] is never clobbered.
pub fn apply(v: &Validated, ws: &Workspace) -> Result<ApplyReport, ToolError> {
    for f in &v.files {
        if ws.resolve(&f.path)? != f.abs {
            return Err(ToolError::Escape { path: f.path.clone() });
        }
        let actual = current_hash(&f.abs, &f.path)?.unwrap_or_else(|| NEW_FILE.to_string());
        if actual != f.before_hash {
            return Err(ToolError::StalePatch { path: f.path.clone(), expected: f.before_hash.clone(), actual });
        }
    }
    let mut created_dirs: Vec<PathBuf> = Vec::new();
    for (i, f) in v.files.iter().enumerate() {
        let result = match &f.after {
            Some(bytes) => create_parents(&f.abs, &mut created_dirs).and_then(|()| write_atomic(&f.abs, bytes, f.mode)),
            None => fs::remove_file(&f.abs),
        };
        if let Err(e) = result {
            let mut rolled_back = true;
            for done in v.files[..i].iter().rev() {
                rolled_back &= restore(&done.abs, &done.before, done.mode);
            }
            for d in created_dirs.iter().rev() {
                rolled_back &= fs::remove_dir(d).is_ok();
            }
            return Err(ToolError::ApplyFailed { path: f.path.clone(), msg: e.to_string(), rolled_back });
        }
    }
    Ok(ApplyReport {
        files: v.files.iter().map(|f| AppliedFile { path: f.path.clone(), before_hash: f.before_hash.clone(), after_hash: f.after_hash.clone(), added: f.added, removed: f.removed }).collect(),
        undo: v
            .files
            .iter()
            .map(|f| Undo { path: f.path.clone(), abs: f.abs.clone(), before: f.before.clone(), mode: f.mode, after_hash: f.after.as_ref().map(|_| f.after_hash.clone()) })
            .collect(),
        created_dirs,
    })
}

/// Creates missing parent directories, recording them outermost-first.
fn create_parents(abs: &Path, created: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut missing = Vec::new();
    let mut p = abs.parent();
    while let Some(d) = p.filter(|d| fs::symlink_metadata(d).is_err()) {
        missing.push(d.to_path_buf());
        p = d.parent();
    }
    for d in missing.into_iter().rev() {
        fs::create_dir(&d)?;
        created.push(d);
    }
    Ok(())
}

/// Undoes an [`apply`] (the repair loop's "put it back"). Refuses — touching
/// nothing — if any file changed since the apply, so later edits are never lost.
pub fn revert(report: &ApplyReport, ws: &Workspace) -> Result<(), ToolError> {
    for u in &report.undo {
        if ws.resolve(&u.path)? != u.abs {
            return Err(ToolError::Escape { path: u.path.clone() });
        }
        let actual = current_hash(&u.abs, &u.path)?;
        if actual != u.after_hash {
            return Err(ToolError::StalePatch {
                path: u.path.clone(),
                expected: u.after_hash.clone().unwrap_or_else(|| DELETED_FILE.into()),
                actual: actual.unwrap_or_else(|| DELETED_FILE.into()),
            });
        }
    }
    for u in report.undo.iter().rev() {
        if !restore(&u.abs, &u.before, u.mode) {
            return Err(io_err(&u.path, std::io::Error::other("could not restore pre-image")));
        }
    }
    for d in report.created_dirs.iter().rev() {
        let _ = fs::remove_dir(d); // only succeeds if still empty; that's intended
    }
    Ok(())
}

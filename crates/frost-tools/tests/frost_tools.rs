use frost_tools::patch::{DELETED_FILE, NEW_FILE};
use frost_tools::{apply, parse_unified_diff, revert, run, validate, DiagnosticsKind, RunSpec, ToolError, Workspace};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ------------------------------------------------------------------ helpers

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!("frost-tools-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn ws_with(files: &[(&str, &str)]) -> (TempDir, Workspace) {
    let t = TempDir::new("ws");
    for (rel, content) in files {
        let p = t.path().join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }
    let ws = Workspace::open(t.path()).unwrap();
    (t, ws)
}

fn numbered(n: usize) -> String {
    (1..=n).map(|i| format!("line {i}\n")).collect()
}

fn spec(argv: &[&str], timeout_ms: u64) -> RunSpec {
    RunSpec {
        argv: argv.iter().map(|s| s.to_string()).collect(),
        cwd: ".".into(),
        timeout: Duration::from_millis(timeout_ms),
        max_output_bytes: 1 << 20,
        env_allowlist: vec!["HOME".into(), "LANG".into(), "TMPDIR".into()],
    }
}

fn no_cancel() -> AtomicBool {
    AtomicBool::new(false)
}

fn read_all(ws: &Workspace, rel: &str) -> String {
    ws.read(rel, 0, 0, usize::MAX).unwrap().content
}

// ------------------------------------------------------------ path safety

#[test]
fn path_escape_attempts_are_rejected() {
    let (t, ws) = ws_with(&[("a.txt", "hi\n"), ("sub/b.txt", "b\n")]);
    assert_eq!(ws.resolve(".").unwrap(), ws.root());
    assert_eq!(ws.resolve("sub/b.txt").unwrap(), ws.root().join("sub/b.txt"));
    assert!(matches!(ws.resolve("../x"), Err(ToolError::Escape { .. })));
    assert!(matches!(ws.resolve("sub/../../x"), Err(ToolError::Escape { .. })));
    assert!(matches!(ws.resolve("sub/../a.txt"), Err(ToolError::Escape { .. })), "any '..' is refused");
    assert!(matches!(ws.resolve("/etc/passwd"), Err(ToolError::AbsolutePath { .. })));
    assert!(matches!(ws.read("/etc/hosts", 0, 0, 100), Err(ToolError::AbsolutePath { .. })));

    // Symlink to a directory outside the root.
    let outside = TempDir::new("outside");
    fs::write(outside.path().join("o.txt"), "outside\n").unwrap();
    symlink(outside.path(), t.path().join("out")).unwrap();
    assert!(matches!(ws.resolve("out"), Err(ToolError::Escape { .. })));
    assert!(matches!(ws.read("out/o.txt", 0, 0, 100), Err(ToolError::Escape { .. })));
    assert!(matches!(ws.list("out", 10), Err(ToolError::Escape { .. })));

    // Dangling symlink pointing outside: writing through it could create a file anywhere.
    symlink("/nonexistent-frost-tools/target-file", t.path().join("dangle")).unwrap();
    assert!(matches!(ws.resolve("dangle"), Err(ToolError::Escape { .. })));

    // Patches and the runner go through the same resolver.
    let p = parse_unified_diff("--- /dev/null\n+++ b/out/new.txt\n@@ -0,0 +1 @@\n+x\n").unwrap();
    let base = BTreeMap::from([("out/new.txt".to_string(), NEW_FILE.to_string())]);
    assert!(matches!(validate(&p, &ws, &base), Err(ToolError::Escape { .. })));
    assert!(!outside.path().join("new.txt").exists());
    let mut s = spec(&["true"], 5_000);
    s.cwd = "out".into();
    assert!(matches!(run(&ws, &s, &no_cancel()), Err(ToolError::Escape { .. })));
    s.cwd = "..".into();
    assert!(matches!(run(&ws, &s, &no_cancel()), Err(ToolError::Escape { .. })));
}

#[test]
fn excluded_files_are_unreadable_unlisted_and_unsearchable() {
    let excluded = [
        ".env",
        ".env.local",
        "key.pem",
        "server.key",
        "id_rsa",
        "id_rsa.pub",
        "aws_credentials.json",
        "my-secret.txt",
        "db.sqlite",
        ".git/config",
        "node_modules/m/index.js",
        "target/debug/out.txt",
    ];
    let mut files: Vec<(&str, &str)> = excluded.iter().map(|p| (*p, "TOKEN=1\n")).collect();
    files.push(("ok.txt", "TOKEN here too\n"));
    let (t, ws) = ws_with(&files);
    for p in excluded {
        assert!(matches!(ws.read(p, 0, 0, 100), Err(ToolError::Excluded { .. })), "{p} must be excluded");
    }
    let names: Vec<String> = ws.list(".", 100).unwrap().entries.into_iter().map(|e| e.name).collect();
    assert_eq!(names, ["ok.txt"]);
    let hits = ws.search("token", 100, true).unwrap().hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].path, "ok.txt");

    // A symlink with an innocent name pointing at a secret is still excluded.
    symlink(t.path().join(".env"), t.path().join("innocent.txt")).unwrap();
    assert!(matches!(ws.read("innocent.txt", 0, 0, 100), Err(ToolError::Excluded { .. })));

    // Writes are excluded too.
    let p = parse_unified_diff("--- /dev/null\n+++ b/.env.production\n@@ -0,0 +1 @@\n+X=1\n").unwrap();
    let base = BTreeMap::from([(".env.production".to_string(), NEW_FILE.to_string())]);
    assert!(matches!(validate(&p, &ws, &base), Err(ToolError::Excluded { .. })));

    // allow_hidden lifts the exclusions (never the escape checks).
    let open = ws.clone().with_allow_hidden(true);
    assert_eq!(read_all(&open, ".env"), "TOKEN=1\n");
    assert!(matches!(open.resolve("../x"), Err(ToolError::Escape { .. })));
}

// --------------------------------------------------------- read-only tools

#[test]
fn read_list_search_and_diagnostics() {
    let long = format!("{}\n", "é".repeat(300));
    let (_t, ws) = ws_with(&[
        ("abc.txt", "abc"),
        ("src/a.rs", "one\ntwo\nthree\nfour\n"),
        ("src/b.rs", "Needle\nneedle\nNEEDLE\n"),
        ("src/long.txt", &long),
        ("bin.dat", "ab\0cd"),
        ("zig/rle.zig", "x"),
        ("zig/rle_test.zig", "x"),
        ("c/b.c", "int b;"),
        ("c/a.c", "int a;"),
        ("c/a.h", ""),
    ]);
    // Known-answer SHA-256 ("abc").
    let r = ws.read("abc.txt", 0, 0, 100).unwrap();
    assert_eq!(r.sha256_hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!((r.total_lines, r.content.as_str()), (1, "abc"));

    let r = ws.read("src/a.rs", 2, 3, 1000).unwrap();
    assert_eq!((r.start_line, r.end_line, r.total_lines, r.content.as_str(), r.truncated), (2, 3, 4, "two\nthree\n", false));
    let r = ws.read("src/a.rs", 1, 0, 6).unwrap();
    assert_eq!((r.end_line, r.content.as_str(), r.truncated), (1, "one\n", true));
    let r = ws.read("src/long.txt", 0, 0, 5).unwrap();
    assert!(r.truncated && r.content.len() <= 5 && r.content.chars().all(|c| c == 'é'));
    assert!(matches!(ws.read("bin.dat", 0, 0, 100), Err(ToolError::NotText { .. })));
    assert!(matches!(ws.read("missing.txt", 0, 0, 100), Err(ToolError::NotFound { .. })));

    let l = ws.list("src", 2).unwrap();
    assert_eq!((l.entries.len(), l.truncated, l.entries[0].name.as_str()), (2, true, "a.rs"));
    assert!(ws.list(".", 100).unwrap().entries.iter().any(|e| e.name == "src" && e.is_dir));

    let s = ws.search("needle", 100, false).unwrap();
    assert_eq!(s.hits.len(), 1);
    assert_eq!((s.hits[0].path.as_str(), s.hits[0].line), ("src/b.rs", 2));
    let s = ws.search("needle", 2, true).unwrap();
    assert_eq!((s.hits.len(), s.truncated), (2, true));
    assert!(ws.search("cd", 10, false).unwrap().hits.is_empty(), "binary files are skipped");
    assert!(matches!(ws.search("", 10, false), Err(ToolError::InvalidArgument { .. })));

    let d = ws.diagnostics(DiagnosticsKind::Cargo, ".").unwrap();
    assert_eq!(d.argv, ["cargo", "check", "--message-format", "short"]);
    assert_eq!(ws.diagnostics(DiagnosticsKind::Zig, "zig").unwrap().argv, ["zig", "test", "--cache-dir", ".zig-cache", "rle_test.zig"]);
    let d = ws.diagnostics(DiagnosticsKind::Clang, "c").unwrap();
    assert_eq!((d.cwd.as_str(), d.argv), ("c", vec!["clang".to_string(), "-fsyntax-only".into(), "a.c".into(), "b.c".into()]));
    assert!(ws.diagnostics(DiagnosticsKind::Clang, "src").is_err());
}

// ------------------------------------------------------------ patch engine

const TWO_FILE_DIFF: &str = "\
diff --git a/f.txt b/f.txt
index 0000000..1111111 100755
--- a/f.txt
+++ b/f.txt
@@ -2,3 +2,3 @@
 line 2
-line 3
+LINE THREE
 line 4
@@ -17,3 +17,4 @@ some fn context
 line 17
-line 18
+LINE EIGHTEEN
+line 18.5
 line 19
diff --git a/g.txt b/g.txt
--- a/g.txt
+++ b/g.txt
@@ -1,2 +1,2 @@
 a
-b
+B
";

#[test]
fn unified_diff_round_trip_then_revert() {
    let original = numbered(20);
    let (t, ws) = ws_with(&[("f.txt", &original), ("g.txt", "a\nb\n")]);
    fs::set_permissions(t.path().join("f.txt"), fs::Permissions::from_mode(0o755)).unwrap();

    let p = parse_unified_diff(TWO_FILE_DIFF).unwrap();
    assert_eq!(p.affected_files(), ["f.txt", "g.txt"]);
    let base = p.base_hashes(&ws).unwrap();
    assert_eq!(base["f.txt"], ws.read("f.txt", 0, 0, 1).unwrap().sha256_hex);
    let v = validate(&p, &ws, &base).unwrap();
    assert!(v.files.iter().all(|f| f.hunk_offsets.iter().all(|&o| o == 0)));
    let report = apply(&v, &ws).unwrap();

    let expected = original.replace("line 3\n", "LINE THREE\n").replace("line 18\n", "LINE EIGHTEEN\nline 18.5\n");
    assert_eq!(read_all(&ws, "f.txt"), expected);
    assert_eq!(read_all(&ws, "g.txt"), "a\nB\n");
    for f in &report.files {
        assert_eq!(f.after_hash, ws.read(&f.path, 0, 0, 1).unwrap().sha256_hex, "{}", f.path);
        assert_eq!(f.before_hash, base[&f.path]);
    }
    assert_eq!((report.files[0].added, report.files[0].removed), (3, 2));
    let mode = fs::metadata(t.path().join("f.txt")).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o755, "mode preserved across atomic write");
    assert!(ws.list(".", 100).unwrap().entries.iter().all(|e| !e.name.contains("frost-tmp")));

    revert(&report, &ws).unwrap();
    assert_eq!(read_all(&ws, "f.txt"), original);
    assert_eq!(read_all(&ws, "g.txt"), "a\nb\n");
    assert_eq!(p.base_hashes(&ws).unwrap(), base);
}

#[test]
fn stale_patch_is_rejected() {
    let (t, ws) = ws_with(&[("f.txt", &numbered(20)), ("g.txt", "a\nb\n")]);
    let p = parse_unified_diff(TWO_FILE_DIFF).unwrap();
    let base = p.base_hashes(&ws).unwrap();

    // Changed between proposal and validation.
    fs::write(t.path().join("g.txt"), "a\nb\nc\n").unwrap();
    match validate(&p, &ws, &base) {
        Err(ToolError::StalePatch { path, expected, actual }) => {
            assert_eq!((path.as_str(), &expected), ("g.txt", &base["g.txt"]));
            assert_ne!(actual, expected);
        }
        other => panic!("expected StalePatch, got {other:?}"),
    }
    assert!(matches!(validate(&p, &ws, &BTreeMap::new()), Err(ToolError::MissingBaseHash { .. })));

    // Changed between validation and apply: nothing is written.
    fs::write(t.path().join("g.txt"), "a\nb\n").unwrap();
    let v = validate(&p, &ws, &base).unwrap();
    fs::write(t.path().join("g.txt"), "a\nb\nedited\n").unwrap();
    assert!(matches!(apply(&v, &ws), Err(ToolError::StalePatch { .. })));
    assert_eq!(read_all(&ws, "f.txt"), numbered(20), "no file touched");
    assert_eq!(read_all(&ws, "g.txt"), "a\nb\nedited\n");

    // Changed after apply: revert refuses rather than clobbering the edit.
    fs::write(t.path().join("g.txt"), "a\nb\n").unwrap();
    let report = apply(&validate(&p, &ws, &base).unwrap(), &ws).unwrap();
    fs::write(t.path().join("g.txt"), "user edit\n").unwrap();
    assert!(matches!(revert(&report, &ws), Err(ToolError::StalePatch { .. })));
    assert_eq!(read_all(&ws, "g.txt"), "user edit\n");
}

fn one_hunk(path: &str, stated_line: usize, ctx: [&str; 3]) -> String {
    format!("--- a/{path}\n+++ b/{path}\n@@ -{stated_line},3 +{stated_line},3 @@\n {}\n-{}\n+CHANGED\n {}\n", ctx[0], ctx[1], ctx[2])
}

#[test]
fn wrong_start_line_located_by_unique_preimage() {
    let (_t, ws) = ws_with(&[("f.txt", &numbered(30))]);
    let base = BTreeMap::from([("f.txt".to_string(), ws.read("f.txt", 0, 0, 1).unwrap().sha256_hex)]);

    // Real position is line 10. Any stated line works while the exact
    // pre-image occurs once in the file; the offset is reported.
    for (stated, offset) in [(10, 0isize), (7, 3), (13, -3), (6, 4), (14, -4), (1, 9), (30, -20)] {
        let p = parse_unified_diff(&one_hunk("f.txt", stated, ["line 10", "line 11", "line 12"])).unwrap();
        let v = validate(&p, &ws, &base).unwrap();
        assert_eq!(v.files[0].hunk_offsets, [offset], "stated line {stated}");
    }
    // Text that doesn't exist at all: rejected.
    let p = parse_unified_diff(&one_hunk("f.txt", 10, ["line 10", "line 99", "line 12"])).unwrap();
    assert!(matches!(validate(&p, &ws, &base), Err(ToolError::HunkMismatch { .. })));

    // Offset matches that are not unique: rejected.
    let (_t2, ws2) = ws_with(&[("r.txt", "a\nb\na\nb\na\nb\n")]);
    let base2 = BTreeMap::from([("r.txt".to_string(), ws2.read("r.txt", 0, 0, 1).unwrap().sha256_hex)]);
    let p = parse_unified_diff("--- a/r.txt\n+++ b/r.txt\n@@ -2,2 +2,2 @@\n a\n-b\n+B\n").unwrap();
    match validate(&p, &ws2, &base2) {
        Err(ToolError::HunkMismatch { msg, .. }) => assert!(msg.contains("ambiguous"), "{msg}"),
        other => panic!("expected ambiguity, got {other:?}"),
    }
}

fn apply_text(ws: &Workspace, diff: &str) -> Result<(), ToolError> {
    let p = parse_unified_diff(diff)?;
    apply(&validate(&p, ws, &p.base_hashes(ws)?)?, ws).map(|_| ())
}

#[test]
fn hunk_header_counts_are_advisory() {
    let (_t, ws) = ws_with(&[("f.txt", &numbered(10)), ("g.txt", "a\nb\nc\n")]);
    // Too many lines claimed (9/9 for a 3-line body) and too few (1/1 for a
    // 4-line body); hunks end at the next '@@', the next file header, and EOF.
    let diff = "\
--- a/f.txt
+++ b/f.txt
@@ -2,9 +2,9 @@
 line 2
-line 3
+LINE 3
@@ -6,1 +6,1 @@
 line 6
-line 7
+LINE 7
+line 7.5
--- a/g.txt
+++ b/g.txt
@@ -1,1 +1,40 @@
 a
-b
+B
 c
";
    apply_text(&ws, diff).unwrap();
    assert_eq!(read_all(&ws, "f.txt"), numbered(10).replace("line 3\n", "LINE 3\n").replace("line 7\n", "LINE 7\nline 7.5\n"));
    assert_eq!(read_all(&ws, "g.txt"), "a\nB\nc\n");
    let p = parse_unified_diff(diff).unwrap();
    assert_eq!((p.files[0].hunks[0].old_count, p.files[0].hunks[0].new_count), (2, 2), "counts recomputed from the body");
    assert_eq!((p.files[0].hunks[1].old_count, p.files[0].hunks[1].new_count), (2, 3));
}

#[test]
fn sloppy_model_diffs_apply_when_the_preimage_is_unique() {
    let (_t, ws) = ws_with(&[("src/lib.rs", "fn a() {}\n\nfn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n"), ("dup.txt", "}\n}\nx\n}\n}\n")]);
    // No a/ b/ prefixes, header without counts, trailer glued to '@@', wrong
    // start line, trailing blank lines after the last hunk.
    let diff = "--- src/lib.rs\n+++ src/lib.rs\n@@ -1 +1 @@fn main() {\n     let x = 1;\n-    println!(\"{x}\");\n+    println!(\"x = {x}\");\n }\n\n\n";
    let p = parse_unified_diff(diff).unwrap();
    assert_eq!(p.affected_files(), ["src/lib.rs"]);
    let v = validate(&p, &ws, &p.base_hashes(&ws).unwrap()).unwrap();
    assert_eq!(v.files[0].hunk_offsets, [3], "found at line 4 although the header said 1");
    apply(&v, &ws).unwrap();
    assert_eq!(read_all(&ws, "src/lib.rs"), "fn a() {}\n\nfn main() {\n    let x = 1;\n    println!(\"x = {x}\");\n}\n");

    // A pre-image that occurs twice, at a wrong stated line, is ambiguous: rejected.
    let before = read_all(&ws, "dup.txt");
    match apply_text(&ws, "--- dup.txt\n+++ dup.txt\n@@ -3,7 +3,1 @@\n }\n-}\n+};\n") {
        Err(ToolError::HunkMismatch { msg, .. }) => assert!(msg.contains("ambiguous"), "{msg}"),
        other => panic!("expected ambiguity, got {other:?}"),
    }
    // Absent pre-image (one byte differs): rejected, file untouched.
    assert!(matches!(apply_text(&ws, "--- dup.txt\n+++ dup.txt\n@@ -1 +1 @@\n }\n-x \n+y\n"), Err(ToolError::HunkMismatch { .. })));
    assert_eq!(read_all(&ws, "dup.txt"), before);

    // A removed line that itself starts with "-- " stays inside the hunk.
    let (_t, ws) = ws_with(&[("m.sql", "select 1;\n-- old comment\nselect 2;\n")]);
    apply_text(&ws, "--- a/m.sql\n+++ b/m.sql\n@@ -1,1 +1,1 @@\n select 1;\n--- old comment\n+-- new comment\n select 2;\n").unwrap();
    assert_eq!(read_all(&ws, "m.sql"), "select 1;\n-- new comment\nselect 2;\n");

    // The real fixture fix, written the way a small model writes it.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/rust-slugify/src/lib.rs");
    let (_t, ws) = ws_with(&[("src/lib.rs", &fs::read_to_string(src).unwrap())]);
    let sloppy = "--- src/lib.rs\n+++ src/lib.rs\n@@ -40,2 +40,9 @@ pub fn slugify(input: &str) -> String {\n     let start = out.find(|c| c != '-').unwrap_or(out.len());\n-    let end = out.rfind(|c| c != '-').unwrap_or(0);\n+    let end = out.rfind(|c| c != '-').map_or(0, |i| i + 1);\n";
    apply_text(&ws, sloppy).unwrap();
    assert!(read_all(&ws, "src/lib.rs").contains("map_or(0, |i| i + 1)"));
}

#[test]
fn new_file_and_delete_file_patches() {
    let (t, ws) = ws_with(&[("old.txt", "bye\nnow\n")]);
    let diff = "\
--- /dev/null
+++ b/sub/dir/new.txt
@@ -0,0 +1,2 @@
+hello
+world
--- a/old.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-bye
-now
";
    let p = parse_unified_diff(diff).unwrap();
    let base = p.base_hashes(&ws).unwrap();
    assert_eq!(base["sub/dir/new.txt"], NEW_FILE);
    let report = apply(&validate(&p, &ws, &base).unwrap(), &ws).unwrap();
    assert_eq!(read_all(&ws, "sub/dir/new.txt"), "hello\nworld\n");
    assert!(!t.path().join("old.txt").exists());
    assert_eq!(report.files[0].before_hash, NEW_FILE);
    assert_eq!(report.files[1].after_hash, DELETED_FILE);
    assert_eq!((report.files[1].added, report.files[1].removed), (0, 2));

    revert(&report, &ws).unwrap();
    assert!(!t.path().join("sub").exists(), "created directories removed on revert");
    assert_eq!(read_all(&ws, "old.txt"), "bye\nnow\n");

    // Creating a file that already exists is stale.
    fs::write(t.path().join("sub.txt"), "x\n").unwrap();
    let p = parse_unified_diff("--- /dev/null\n+++ b/sub.txt\n@@ -0,0 +1 @@\n+y\n").unwrap();
    let base = BTreeMap::from([("sub.txt".to_string(), NEW_FILE.to_string())]);
    assert!(matches!(validate(&p, &ws, &base), Err(ToolError::StalePatch { .. })));
    // A deletion that leaves lines behind is refused.
    let p = parse_unified_diff("--- a/old.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-bye\n").unwrap();
    assert!(matches!(validate(&p, &ws, &p.base_hashes(&ws).unwrap()), Err(ToolError::HunkMismatch { .. })));
}

#[test]
fn no_newline_at_end_of_file_is_honoured() {
    let (_t, ws) = ws_with(&[("n.txt", "a\nb")]);
    let base = BTreeMap::from([("n.txt".to_string(), ws.read("n.txt", 0, 0, 1).unwrap().sha256_hex)]);
    let add_eol = "--- a/n.txt\n+++ b/n.txt\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+b\n";
    apply(&validate(&parse_unified_diff(add_eol).unwrap(), &ws, &base).unwrap(), &ws).unwrap();
    assert_eq!(read_all(&ws, "n.txt"), "a\nb\n");

    // A missing marker is tolerated and the file keeps its (absent) final newline.
    let (_t, ws) = ws_with(&[("n.txt", "a\nb")]);
    let ignores = "--- a/n.txt\n+++ b/n.txt\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n";
    apply(&validate(&parse_unified_diff(ignores).unwrap(), &ws, &base).unwrap(), &ws).unwrap();
    assert_eq!(read_all(&ws, "n.txt"), "a\nc");

    // A marker that is present but false (the file does end in '\n') is still refused.
    let (_t, ws) = ws_with(&[("n.txt", "a\nb\n")]);
    let base = BTreeMap::from([("n.txt".to_string(), ws.read("n.txt", 0, 0, 1).unwrap().sha256_hex)]);
    let lies = "--- a/n.txt\n+++ b/n.txt\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+c\n";
    assert!(matches!(validate(&parse_unified_diff(lies).unwrap(), &ws, &base), Err(ToolError::HunkMismatch { .. })));
}

#[test]
fn parser_rejects_renames_binary_and_bad_counts() {
    let unsupported = [
        "--- a/x\n+++ b/y\n@@ -1 +1 @@\n-a\n+b\n",
        "diff --git a/x b/y\nsimilarity index 90%\nrename from x\nrename to y\n",
        "diff --git a/x b/x\nBinary files a/x and b/x differ\n",
        "diff --git a/x b/x\nGIT binary patch\nliteral 1\n",
    ];
    for d in unsupported {
        assert!(matches!(parse_unified_diff(d), Err(ToolError::Unsupported { .. })), "{d:?}");
    }
    let malformed = [
        "",
        "just prose\n",
        "--- a/x\n@@ -1 +1 @@\n",                                 // missing +++
        "--- a/x\n+++ b/x\n@@ -1 +1\n-a\n+b\n",                     // no closing @@
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\nplain text\n+b\n",      // not a hunk line
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n@@ -2 +2 @@\n-a\n+b\n",     // empty hunk
        "--- a/x\n+++ b/x\n",                                       // no hunks
        "--- /dev/null\n+++ b/x\n@@ -1,0 +1 @@\n+a\n",              // creation not at 0,0
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", // duplicate
    ];
    for d in malformed {
        assert!(matches!(parse_unified_diff(d), Err(ToolError::Parse { .. })), "{d:?}");
    }
}

#[test]
fn apply_is_all_or_nothing() {
    let (t, ws) = ws_with(&[("a.txt", "a\n"), ("b/b.txt", "b\n")]);
    let diff = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+A\n--- a/b/b.txt\n+++ b/b/b.txt\n@@ -1 +1 @@\n-b\n+B\n";
    let p = parse_unified_diff(diff).unwrap();
    let v = validate(&p, &ws, &p.base_hashes(&ws).unwrap()).unwrap();
    // The second write fails (read-only directory): the first must be rolled back.
    let bdir = t.path().join("b");
    fs::set_permissions(&bdir, fs::Permissions::from_mode(0o555)).unwrap();
    let result = apply(&v, &ws);
    fs::set_permissions(&bdir, fs::Permissions::from_mode(0o755)).unwrap();
    match result {
        Err(ToolError::ApplyFailed { path, rolled_back, .. }) => assert_eq!((path.as_str(), rolled_back), ("b/b.txt", true)),
        other => panic!("expected ApplyFailed, got {other:?}"),
    }
    assert_eq!(read_all(&ws, "a.txt"), "a\n");
    assert_eq!(read_all(&ws, "b/b.txt"), "b\n");
    assert!(ws.list(".", 100).unwrap().entries.iter().all(|e| !e.name.contains("frost-tmp")));
}

// ------------------------------------------------------------------ runner

#[test]
fn runner_reports_exit_codes_and_resolves_argv0() {
    let (_t, ws) = ws_with(&[]);
    let r = run(&ws, &spec(&["true"], 5_000), &no_cancel()).unwrap();
    assert_eq!((r.exit_code, r.signal, r.timed_out, r.cancelled), (Some(0), None, false, false));
    assert!(r.argv[0].starts_with('/') && r.argv[0].ends_with("/true"));
    assert_eq!(r.cwd, ws.root().display().to_string());
    let r = run(&ws, &spec(&["false"], 5_000), &no_cancel()).unwrap();
    assert_eq!(r.exit_code, Some(1));
    let r = run(&ws, &spec(&["/bin/sh", "-c", "exit 7"], 5_000), &no_cancel()).unwrap();
    assert_eq!(r.exit_code, Some(7));
}

#[test]
fn runner_timeout_kills_within_bound() {
    let (_t, ws) = ws_with(&[]);
    let timeout_ms = 500;
    let r = run(&ws, &spec(&["sleep", "30"], timeout_ms), &no_cancel()).unwrap();
    assert!(r.timed_out && !r.cancelled);
    assert_eq!((r.exit_code, r.signal), (None, Some(15)), "SIGTERM delivered to the group");
    assert!(r.duration_ms < 2_500, "took {} ms", r.duration_ms);
    println!("timeout-to-kill latency: {} ms (duration {} ms, timeout {timeout_ms} ms)", r.duration_ms - timeout_ms, r.duration_ms);
}

#[test]
fn runner_cancel_from_another_thread() {
    let (_t, ws) = ws_with(&[]);
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    let setter = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        c2.store(true, Ordering::Relaxed);
    });
    let started = Instant::now();
    let r = run(&ws, &spec(&["sleep", "30"], 60_000), &cancel).unwrap();
    setter.join().unwrap();
    assert!(r.cancelled && !r.timed_out);
    assert_eq!(r.signal, Some(15));
    assert!(started.elapsed() < Duration::from_millis(2_500), "{:?}", started.elapsed());

    // Already cancelled: nothing is spawned.
    let r = run(&ws, &spec(&["sleep", "30"], 60_000), &cancel).unwrap();
    assert!(r.cancelled && r.exit_code.is_none() && r.duration_ms == 0);
}

#[test]
fn runner_caps_output_keeping_head_and_tail() {
    let (_t, ws) = ws_with(&[]);
    let mut s = spec(&["/bin/dd", "if=/dev/zero", "bs=1m", "count=5"], 60_000);
    s.max_output_bytes = 64 * 1024;
    let r = run(&ws, &s, &no_cancel()).unwrap();
    assert_eq!(r.exit_code, Some(0));
    assert!(r.stdout_truncated && !r.stderr_truncated);
    let dropped = 5 * 1024 * 1024 - 64 * 1024;
    let marker = format!("\n…[truncated {dropped} bytes]…\n");
    assert!(r.stdout.contains(&marker));
    assert_eq!(r.stdout.len(), 64 * 1024 + marker.len());
    assert!(r.stdout.starts_with('\0') && r.stdout.ends_with('\0'));
    assert!(r.stderr.contains("records"), "dd stats on stderr: {}", r.stderr);
}

#[test]
fn runner_scrubs_environment() {
    let (_t, ws) = ws_with(&[]);
    let mut s = spec(&["/usr/bin/env"], 5_000);
    s.env_allowlist = vec!["HOME".into(), "LANG".into(), "FROST_TOOLS_SURELY_UNSET_VAR".into()];
    let r = run(&ws, &s, &no_cancel()).unwrap();
    let printed: Vec<&str> = r.stdout.lines().filter_map(|l| l.split_once('=').map(|(k, _)| k)).collect();
    for k in &printed {
        assert!(["HOME", "LANG", "PATH"].contains(k), "unexpected env var {k} leaked: {}", r.stdout);
    }
    let mut sorted = printed.clone();
    sorted.sort();
    assert_eq!(sorted, r.env_keys, "env_keys records exactly what the child saw");
    let path = r.stdout.lines().find_map(|l| l.strip_prefix("PATH=")).unwrap();
    assert!(path.starts_with("/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"), "safe default PATH, got {path}");
    assert!(run(&ws, &spec(&["env"], 5_000), &no_cancel()).is_ok(), "bare name resolves on the default PATH");
    let mut bad = spec(&["true"], 5_000);
    bad.env_allowlist = vec!["A=B".into()];
    assert!(matches!(run(&ws, &bad, &no_cancel()), Err(ToolError::BadCommand { .. })));
}

#[test]
fn runner_never_uses_a_shell() {
    let (t, ws) = ws_with(&[]);
    let r = run(&ws, &spec(&["echo", "$HOME"], 5_000), &no_cancel()).unwrap();
    assert_eq!(r.stdout, "$HOME\n");
    let r = run(&ws, &spec(&["echo", "a;", "touch", "pwned", "&&", "`id`", ">", "out"], 5_000), &no_cancel()).unwrap();
    assert_eq!(r.stdout, "a; touch pwned && `id` > out\n");
    assert!(!t.path().join("pwned").exists() && !t.path().join("out").exists());
    for argv0 in ["bin/echo", "../echo", "echo; ls", ""] {
        assert!(matches!(run(&ws, &spec(&[argv0], 5_000), &no_cancel()), Err(ToolError::BadCommand { .. })), "{argv0:?}");
    }
    assert!(matches!(run(&ws, &spec(&[], 5_000), &no_cancel()), Err(ToolError::BadCommand { .. })));
}

fn process_gone(marker: &str) -> bool {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let found = std::process::Command::new("/usr/bin/pgrep").args(["-f", marker]).output().unwrap().status.success();
        if !found {
            return true;
        }
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn runner_kills_the_whole_process_group() {
    let (_t, ws) = ws_with(&[]);
    // Timeout path: both the background and foreground sleeps must die.
    let r = run(&ws, &spec(&["sh", "-c", "sleep 37.4242 & sleep 37.4242"], 300), &no_cancel()).unwrap();
    assert!(r.timed_out && r.duration_ms < 2_500);
    assert!(process_gone("sleep 37.4242"), "child tree survived the timeout");

    // Normal exit: a background job holding stdout open neither outlives the
    // command nor stalls the pipe readers.
    let r = run(&ws, &spec(&["sh", "-c", "sleep 38.5151 & echo started"], 10_000), &no_cancel()).unwrap();
    assert_eq!((r.exit_code, r.stdout.as_str()), (Some(0), "started\n"));
    assert!(r.duration_ms < 2_500, "took {} ms", r.duration_ms);
    assert!(process_gone("sleep 38.5151"), "background job outlived the command");
}

// ------------------------------------------------------- fixture oracles

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dest);
        } else {
            fs::copy(e.path(), dest).unwrap();
        }
    }
}

fn argv_of(v: &serde_json::Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn run_fixture_cmd(ws: &Workspace, argv: Vec<String>) -> frost_tools::RunResult {
    let s = RunSpec {
        argv,
        cwd: ".".into(),
        timeout: Duration::from_secs(300),
        max_output_bytes: 256 * 1024,
        // PATH deliberately omitted: exercises the safe default PATH.
        env_allowlist: ["HOME", "CARGO_HOME", "RUSTUP_HOME", "TMPDIR", "LANG"].map(String::from).to_vec(),
    };
    run(ws, &s, &no_cancel()).unwrap()
}

// Known-correct fixes, kept here (outside the fixtures) so the assistant never sees them.
const RUST_FIX: &str = r#"--- a/src/lib.rs
+++ b/src/lib.rs
@@ -13,5 +13,5 @@
     // Trim the separator runs at both ends.
     let start = out.find(|c| c != '-').unwrap_or(out.len());
-    let end = out.rfind(|c| c != '-').unwrap_or(0);
+    let end = out.rfind(|c| c != '-').map_or(0, |i| i + 1);
     if start >= end {
         return String::new();
"#;
const ZIG_FIX: &str = r#"--- a/rle.zig
+++ b/rle.zig
@@ -12,3 +12,3 @@
         var run: usize = 1;
-        while (i + run < input.len - 1 and input[i + run] == byte and run < 255) : (run += 1) {}
+        while (i + run < input.len and input[i + run] == byte and run < 255) : (run += 1) {}
         if (written + 2 > out.len) return error.OutputTooSmall;
"#;
const C_FIX: &str = r#"--- a/kv.c
+++ b/kv.c
@@ -12,3 +12,3 @@
     size_t value_len = strcspn(value, "\r\n");
-    if (key_len >= sizeof out->key || value_len > sizeof out->value) {
+    if (key_len >= sizeof out->key || value_len >= sizeof out->value) {
         return -1;
"#;

/// The oracle is live: each fixture builds, FAILS its held-out test as
/// shipped, and passes once the correct fix goes through the patch engine.
/// Runs on temp copies; the fixtures themselves are never modified.
fn check_fixture(name: &str, fix: &str) {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(name);
    let tmp = TempDir::new(name);
    copy_dir(&src, tmp.path());
    let ws = Workspace::open(tmp.path()).unwrap();
    let expected: serde_json::Value = serde_json::from_str(&read_all(&ws, "EXPECTED.json")).unwrap();
    assert_eq!(expected["pass_criteria"], "exit 0");
    assert!(read_all(&ws, "TASK.md").len() > 40);
    let held_out = argv_of(&expected["held_out"]);
    for h in &held_out {
        ws.read(h, 0, 0, 1).unwrap(); // exists and is readable
    }

    let build = run_fixture_cmd(&ws, argv_of(&expected["build"]));
    assert_eq!(build.exit_code, Some(0), "{name} build failed:\n{}\n{}", build.stdout, build.stderr);
    let test = run_fixture_cmd(&ws, argv_of(&expected["test"]));
    assert!(!test.timed_out, "{name} test timed out");
    assert_ne!(test.exit_code, Some(0), "{name}: held-out test must FAIL before the fix:\n{}\n{}", test.stdout, test.stderr);

    let patch = parse_unified_diff(fix).unwrap();
    assert!(patch.affected_files().iter().all(|f| !held_out.contains(f)), "fix must not touch held-out files");
    apply(&validate(&patch, &ws, &patch.base_hashes(&ws).unwrap()).unwrap(), &ws).unwrap();
    let build = run_fixture_cmd(&ws, argv_of(&expected["build"]));
    assert_eq!(build.exit_code, Some(0), "{name} build after fix:\n{}", build.stderr);
    let test = run_fixture_cmd(&ws, argv_of(&expected["test"]));
    assert_eq!(test.exit_code, Some(0), "{name}: held-out test must PASS after the fix:\n{}\n{}", test.stdout, test.stderr);
}

#[test]
fn fixture_rust_slugify_oracle_is_live() {
    check_fixture("rust-slugify", RUST_FIX);
}

#[test]
fn fixture_zig_rle_oracle_is_live() {
    check_fixture("zig-rle", ZIG_FIX);
}

#[test]
fn fixture_c_parse_kv_oracle_is_live() {
    check_fixture("c-parse-kv", C_FIX);
}

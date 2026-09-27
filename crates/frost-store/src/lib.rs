//! SQLite persistence for conversations, repo index metadata, embedding cache, attempts.
//!
//! [`Store`] owns a single connection: it is `Send` but not `Sync`, so the app wraps it in a
//! `Mutex`. Ids are 32 lowercase hex chars, timestamps are unix milliseconds, vectors are
//! little-endian `f32` blobs, and `meta`/`payload` columns hold JSON text.

use std::hash::{BuildHasher, Hasher, RandomState};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, Type, ValueRef};
use rusqlite::{ffi, params, Connection, OptionalExtension, Params, Row, ToSql, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("{0} not found: {1}")]
    NotFound(&'static str, String),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid role: {0:?}")]
    InvalidRole(String),
    #[error("corrupt store: {0}")]
    Corrupt(String),
}

pub type Result<T, E = StoreError> = std::result::Result<T, E>;

/// A closed set of strings stored as TEXT: one list drives serde, `FromStr`, and SQL conversion.
macro_rules! str_enum {
    ($(#[$m:meta])* $name:ident, $err:expr, { $($var:ident => $s:literal),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name { $(#[serde(rename = $s)] $var),+ }

        impl $name {
            pub fn as_str(self) -> &'static str {
                match self { $(Self::$var => $s),+ }
            }
        }

        impl std::str::FromStr for $name {
            type Err = StoreError;
            fn from_str(s: &str) -> Result<Self> {
                match s { $($s => Ok(Self::$var),)+ _ => Err(($err)(s)) }
            }
        }

        impl ToSql for $name {
            fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
                Ok(self.as_str().into())
            }
        }

        impl FromSql for $name {
            fn column_result(v: ValueRef<'_>) -> FromSqlResult<Self> {
                v.as_str()?.parse().map_err(|e: StoreError| FromSqlError::Other(Box::new(e)))
            }
        }
    };
}

str_enum!(
    /// Author of a chat message.
    Role, |s: &str| StoreError::InvalidRole(s.to_owned()), {
        System => "system", User => "user", Assistant => "assistant", Tool => "tool",
    }
);

str_enum!(
    AttemptKind, |s: &str| StoreError::Corrupt(format!("unknown attempt kind {s:?}")), {
        ProposeDiff => "propose_diff", ApplyDiff => "apply_diff", RunCommand => "run_command",
        Repair => "repair",
    }
);

str_enum!(
    AttemptStatus, |s: &str| StoreError::Corrupt(format!("unknown attempt status {s:?}")), {
        Proposed => "proposed", Approved => "approved", Denied => "denied", Applied => "applied",
        Passed => "passed", Failed => "failed", Timeout => "timeout", Cancelled => "cancelled",
    }
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub mode: String,
    pub system_prompt: String,
    pub repo_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub seq: i64,
    pub role: Role,
    pub content: String,
    pub created_at: i64,
    /// Generator id, quantization, backend, token counts, finish_reason, cancelled, citations,
    /// tool calls — schemaless on purpose.
    pub meta: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Repo {
    pub id: String,
    pub path: String,
    pub added_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexGeneration {
    pub repo_id: String,
    pub generation: i64,
    pub created_at: i64,
    pub finished_at: Option<i64>,
    pub vector_file: String,
    pub chunk_count: i64,
    pub recipe: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewChunk {
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub digest: Vec<u8>,
    pub content: String,
    pub symbol: Option<String>,
    pub kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: i64,
    pub repo_id: String,
    pub generation: i64,
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub digest: Vec<u8>,
    pub content: String,
    pub symbol: Option<String>,
    pub kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub id: String,
    pub conversation_id: String,
    pub message_id: Option<String>,
    pub kind: AttemptKind,
    pub payload: Value,
    pub status: AttemptStatus,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Forward-only; entry `i` upgrades the schema from version `i` to `i + 1`. Never edit a
/// shipped entry, append a new one.
const MIGRATIONS: &[&str] = &[r#"
CREATE TABLE conversations(
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    mode TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    repo_path TEXT NULL
);
CREATE TABLE messages(
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('system','user','assistant','tool')),
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    UNIQUE(conversation_id, seq)
);
CREATE TABLE repos(
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at INTEGER NOT NULL
);
CREATE TABLE index_generations(
    repo_id TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    finished_at INTEGER NULL,
    vector_file TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    recipe TEXT NOT NULL,
    PRIMARY KEY(repo_id, generation)
);
CREATE TABLE chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    digest BLOB NOT NULL,
    content TEXT NOT NULL,
    symbol TEXT NULL,
    kind TEXT NULL,
    FOREIGN KEY(repo_id, generation) REFERENCES index_generations(repo_id, generation) ON DELETE CASCADE
);
CREATE INDEX chunks_generation ON chunks(repo_id, generation);
CREATE INDEX chunks_digest ON chunks(digest);
CREATE TABLE embedding_cache(
    recipe TEXT NOT NULL,
    digest BLOB NOT NULL,
    vector BLOB NOT NULL,
    bytes INTEGER NOT NULL,
    last_used INTEGER NOT NULL,
    PRIMARY KEY(recipe, digest)
);
CREATE INDEX embedding_cache_last_used ON embedding_cache(last_used);
CREATE TABLE attempts(
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    exit_code INTEGER NULL,
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
-- FK child index: keeps cascade deletes and list_attempts off a full scan.
CREATE INDEX attempts_conversation ON attempts(conversation_id);
CREATE TABLE settings(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#];

const CONVERSATION: &str =
    "SELECT id, title, created_at, updated_at, mode, system_prompt, repo_path FROM conversations";
const MESSAGE: &str =
    "SELECT id, conversation_id, seq, role, content, created_at, meta FROM messages";
const GENERATION: &str = "SELECT repo_id, generation, created_at, finished_at, vector_file, chunk_count, recipe FROM index_generations";
const CHUNK: &str = "SELECT id, repo_id, generation, path, start_line, end_line, digest, content, symbol, kind FROM chunks";
const ATTEMPT: &str = "SELECT id, conversation_id, message_id, kind, payload, status, exit_code, stdout, stderr, duration_ms, created_at, updated_at FROM attempts";

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        Self::init(Connection::open(path)?)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(mut conn: Connection) -> Result<Self> {
        conn.busy_timeout(Duration::from_millis(5000))?;
        conn.pragma_update(None, "foreign_keys", true)?;
        // File databases report "wal"; in-memory ones report "memory" and that's fine.
        let _: String = conn.pragma_update_and_check(None, "journal_mode", "WAL", |r| r.get(0))?;
        migrate(&mut conn)?;
        Ok(Self { conn })
    }

    // ---- conversations -------------------------------------------------------------------

    pub fn create_conversation(&self, title: &str, mode: &str, system_prompt: &str) -> Result<Conversation> {
        let now = now_ms();
        let c = Conversation {
            id: new_id(),
            title: title.into(),
            created_at: now,
            updated_at: now,
            mode: mode.into(),
            system_prompt: system_prompt.into(),
            repo_path: None,
        };
        self.conn.execute(
            "INSERT INTO conversations(id, title, created_at, updated_at, mode, system_prompt, repo_path)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![c.id, c.title, c.created_at, c.updated_at, c.mode, c.system_prompt, c.repo_path],
        )?;
        Ok(c)
    }

    pub fn list_conversations(&self) -> Result<Vec<Conversation>> {
        self.all(&format!("{CONVERSATION} ORDER BY updated_at DESC, rowid DESC"), [], conversation)
    }

    pub fn get_conversation(&self, id: &str) -> Result<Conversation> {
        self.conn
            .query_row(&format!("{CONVERSATION} WHERE id = ?1"), [id], conversation)
            .optional()?
            .ok_or_else(|| StoreError::NotFound("conversation", id.into()))
    }

    pub fn rename_conversation(&self, id: &str, title: &str) -> Result<()> {
        let n = self.conn.execute("UPDATE conversations SET title = ?2 WHERE id = ?1", [id, title])?;
        changed(n, "conversation", id)
    }

    pub fn set_repo_path(&self, id: &str, repo_path: Option<&str>) -> Result<()> {
        let n = self
            .conn
            .execute("UPDATE conversations SET repo_path = ?2 WHERE id = ?1", params![id, repo_path])?;
        changed(n, "conversation", id)
    }

    pub fn set_mode(&self, id: &str, mode: &str) -> Result<()> {
        let n = self.conn.execute("UPDATE conversations SET mode = ?2 WHERE id = ?1", [id, mode])?;
        changed(n, "conversation", id)
    }

    /// Hard delete; messages and attempts go with it via `ON DELETE CASCADE`.
    pub fn delete_conversation(&self, id: &str) -> Result<()> {
        let n = self.conn.execute("DELETE FROM conversations WHERE id = ?1", [id])?;
        changed(n, "conversation", id)
    }

    // ---- messages ------------------------------------------------------------------------

    /// Appends at `max(seq) + 1` and bumps the conversation's `updated_at`, atomically.
    pub fn append_message(&mut self, conversation_id: &str, role: Role, content: &str, meta: Value) -> Result<Message> {
        self.append_raw(conversation_id, role.as_str(), content, meta)
    }

    /// Takes the role as text so the schema CHECK (and the rollback it forces) stays testable.
    fn append_raw(&mut self, conversation_id: &str, role: &str, content: &str, meta: Value) -> Result<Message> {
        let meta_json = serde_json::to_string(&meta)?;
        let (id, now) = (new_id(), now_ms());
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let n = tx.execute("UPDATE conversations SET updated_at = ?2 WHERE id = ?1", params![conversation_id, now])?;
        changed(n, "conversation", conversation_id)?;
        let seq: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?1",
            [conversation_id],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO messages(id, conversation_id, seq, role, content, created_at, meta)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![id, conversation_id, seq, role, content, now, meta_json],
        )
        .map_err(|e| match &e {
            rusqlite::Error::SqliteFailure(f, _) if f.extended_code == ffi::SQLITE_CONSTRAINT_CHECK => {
                StoreError::InvalidRole(role.into())
            }
            _ => e.into(),
        })?;
        tx.commit()?;
        Ok(Message {
            id,
            conversation_id: conversation_id.into(),
            seq,
            role: role.parse()?,
            content: content.into(),
            created_at: now,
            meta,
        })
    }

    pub fn update_message(&self, id: &str, content: &str, meta: Value) -> Result<()> {
        let meta = serde_json::to_string(&meta)?;
        let n = self
            .conn
            .execute("UPDATE messages SET content = ?2, meta = ?3 WHERE id = ?1", params![id, content, meta])?;
        changed(n, "message", id)
    }

    pub fn list_messages(&self, conversation_id: &str) -> Result<Vec<Message>> {
        self.all(&format!("{MESSAGE} WHERE conversation_id = ?1 ORDER BY seq"), [conversation_id], message)
    }

    /// Deletes every message with `seq > seq` (edit/regenerate branching). Returns the count.
    pub fn truncate_after(&self, conversation_id: &str, seq: i64) -> Result<usize> {
        Ok(self
            .conn
            .execute("DELETE FROM messages WHERE conversation_id = ?1 AND seq > ?2", params![conversation_id, seq])?)
    }

    pub fn last_message(&self, conversation_id: &str) -> Result<Option<Message>> {
        Ok(self
            .conn
            .query_row(
                &format!("{MESSAGE} WHERE conversation_id = ?1 ORDER BY seq DESC LIMIT 1"),
                [conversation_id],
                message,
            )
            .optional()?)
    }

    // ---- repos ---------------------------------------------------------------------------

    /// Returns the existing row when `path` is already registered.
    pub fn upsert_repo(&self, path: &str) -> Result<Repo> {
        Ok(self.conn.query_row(
            "INSERT INTO repos(id, path, added_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(path) DO UPDATE SET path = excluded.path
             RETURNING id, path, added_at",
            params![new_id(), path, now_ms()],
            repo,
        )?)
    }

    pub fn list_repos(&self) -> Result<Vec<Repo>> {
        self.all("SELECT id, path, added_at FROM repos ORDER BY added_at, rowid", [], repo)
    }

    /// Cascades to generations and their chunks.
    pub fn remove_repo(&self, id: &str) -> Result<()> {
        let n = self.conn.execute("DELETE FROM repos WHERE id = ?1", [id])?;
        changed(n, "repo", id)
    }

    // ---- index generations & chunks ------------------------------------------------------

    /// Starts generation `max + 1` (single statement, so atomic). Unfinished until `finish_generation`.
    pub fn begin_generation(&self, repo_id: &str, vector_file: &str, recipe: &str) -> Result<i64> {
        self.conn
            .query_row(
                "INSERT INTO index_generations(repo_id, generation, created_at, vector_file, recipe)
                 SELECT ?1, COALESCE(MAX(generation), 0) + 1, ?2, ?3, ?4
                 FROM index_generations WHERE repo_id = ?1
                 RETURNING generation",
                params![repo_id, now_ms(), vector_file, recipe],
                |r| r.get(0),
            )
            .map_err(|e| fk(e, "repo", repo_id))
    }

    /// One transaction for the whole batch; ids come back in input order.
    pub fn insert_chunks(&mut self, repo_id: &str, generation: i64, chunks: &[NewChunk]) -> Result<Vec<i64>> {
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut ids = Vec::with_capacity(chunks.len());
        {
            let mut st = tx.prepare_cached(
                "INSERT INTO chunks(repo_id, generation, path, start_line, end_line, digest, content, symbol, kind)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            for c in chunks {
                let id = st
                    .insert(params![repo_id, generation, c.path, c.start_line, c.end_line, c.digest, c.content, c.symbol, c.kind])
                    .map_err(|e| fk(e, "generation", &format!("{repo_id}#{generation}")))?;
                ids.push(id);
            }
        }
        tx.commit()?;
        Ok(ids)
    }

    pub fn finish_generation(&self, repo_id: &str, generation: i64, chunk_count: i64) -> Result<()> {
        let n = self.conn.execute(
            "UPDATE index_generations SET finished_at = ?3, chunk_count = ?4 WHERE repo_id = ?1 AND generation = ?2",
            params![repo_id, generation, now_ms(), chunk_count],
        )?;
        changed(n, "generation", &format!("{repo_id}#{generation}"))
    }

    pub fn latest_finished_generation(&self, repo_id: &str) -> Result<Option<IndexGeneration>> {
        Ok(self
            .conn
            .query_row(
                &format!("{GENERATION} WHERE repo_id = ?1 AND finished_at IS NOT NULL ORDER BY generation DESC LIMIT 1"),
                [repo_id],
                generation,
            )
            .optional()?)
    }

    /// Keeps the newest `keep_latest` finished generations; their chunks cascade away.
    /// Unfinished generations newer than the latest finished one are in-progress builds and
    /// are never pruned; unfinished ones older than it are abandoned and are.
    pub fn prune_generations(&self, repo_id: &str, keep_latest: usize) -> Result<usize> {
        Ok(self.conn.execute(
            "DELETE FROM index_generations
             WHERE repo_id = ?1
               AND generation NOT IN (
                   SELECT generation FROM index_generations
                   WHERE repo_id = ?1 AND finished_at IS NOT NULL
                   ORDER BY generation DESC LIMIT ?2)
               AND (finished_at IS NOT NULL OR generation < (
                   SELECT COALESCE(MAX(generation), 0) FROM index_generations
                   WHERE repo_id = ?1 AND finished_at IS NOT NULL))",
            params![repo_id, i64::try_from(keep_latest).unwrap_or(i64::MAX)],
        )?)
    }

    pub fn chunks_for_generation(&self, repo_id: &str, generation: i64) -> Result<Vec<Chunk>> {
        self.all(&format!("{CHUNK} WHERE repo_id = ?1 AND generation = ?2 ORDER BY id"), params![repo_id, generation], chunk)
    }

    /// Preserves the requested order (and duplicates); ids that no longer exist are skipped.
    pub fn get_chunks(&self, ids: &[i64]) -> Result<Vec<Chunk>> {
        let mut st = self.conn.prepare_cached(&format!("{CHUNK} WHERE id = ?1"))?;
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(c) = st.query_row([id], chunk).optional()? {
                out.push(c);
            }
        }
        Ok(out)
    }

    /// Across every generation of the repo, newest first.
    pub fn find_chunks_by_digest(&self, repo_id: &str, digest: &[u8]) -> Result<Vec<Chunk>> {
        self.all(
            &format!("{CHUNK} WHERE repo_id = ?1 AND digest = ?2 ORDER BY generation DESC, id"),
            params![repo_id, digest],
            chunk,
        )
    }

    // ---- embedding cache -----------------------------------------------------------------

    /// A hit refreshes `last_used` in the same statement.
    pub fn get_embedding(&self, recipe: &str, digest: &[u8]) -> Result<Option<Vec<f32>>> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row(
                "UPDATE embedding_cache SET last_used = ?3 WHERE recipe = ?1 AND digest = ?2 RETURNING vector",
                params![recipe, digest, now_ms()],
                |r| r.get(0),
            )
            .optional()?;
        blob.map(|b| decode_f32(&b)).transpose()
    }

    pub fn put_embedding(&self, recipe: &str, digest: &[u8], vector: &[f32]) -> Result<()> {
        let blob: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO embedding_cache(recipe, digest, vector, bytes, last_used) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(recipe, digest) DO UPDATE
             SET vector = excluded.vector, bytes = excluded.bytes, last_used = excluded.last_used",
            params![recipe, digest, blob, blob.len() as i64, now_ms()],
        )?;
        Ok(())
    }

    /// Sum of vector blob sizes (keys and SQLite overhead not counted).
    pub fn embedding_cache_bytes(&self) -> Result<u64> {
        let n: i64 = self.conn.query_row("SELECT COALESCE(SUM(bytes), 0) FROM embedding_cache", [], |r| r.get(0))?;
        Ok(n as u64)
    }

    /// Evicts least-recently-used entries until the total is `<= max_bytes`. Returns the count.
    pub fn prune_embeddings(&self, max_bytes: u64) -> Result<usize> {
        // Running total from most- to least-recent: everything past the budget goes.
        Ok(self.conn.execute(
            "DELETE FROM embedding_cache WHERE rowid IN (
                 SELECT rowid FROM (
                     SELECT rowid, SUM(bytes) OVER (ORDER BY last_used DESC, rowid DESC) AS running
                     FROM embedding_cache)
                 WHERE running > ?1)",
            [i64::try_from(max_bytes).unwrap_or(i64::MAX)],
        )?)
    }

    // ---- attempts ------------------------------------------------------------------------

    /// New attempts start as `Proposed`.
    pub fn insert_attempt(
        &self,
        conversation_id: &str,
        message_id: Option<&str>,
        kind: AttemptKind,
        payload: Value,
    ) -> Result<Attempt> {
        let now = now_ms();
        let a = Attempt {
            id: new_id(),
            conversation_id: conversation_id.into(),
            message_id: message_id.map(Into::into),
            kind,
            payload,
            status: AttemptStatus::Proposed,
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            duration_ms: None,
            created_at: now,
            updated_at: now,
        };
        self.conn
            .execute(
                "INSERT INTO attempts(id, conversation_id, message_id, kind, payload, status, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    a.id,
                    a.conversation_id,
                    a.message_id,
                    a.kind,
                    serde_json::to_string(&a.payload)?,
                    a.status,
                    a.created_at,
                    a.updated_at
                ],
            )
            .map_err(|e| fk(e, "conversation", conversation_id))?;
        Ok(a)
    }

    pub fn set_attempt_status(&self, id: &str, status: AttemptStatus) -> Result<()> {
        let n = self
            .conn
            .execute("UPDATE attempts SET status = ?2, updated_at = ?3 WHERE id = ?1", params![id, status, now_ms()])?;
        changed(n, "attempt", id)
    }

    pub fn finish_attempt(
        &self,
        id: &str,
        status: AttemptStatus,
        exit_code: Option<i32>,
        stdout: &str,
        stderr: &str,
        duration_ms: Option<i64>,
    ) -> Result<()> {
        let n = self.conn.execute(
            "UPDATE attempts SET status = ?2, exit_code = ?3, stdout = ?4, stderr = ?5, duration_ms = ?6, updated_at = ?7
             WHERE id = ?1",
            params![id, status, exit_code, stdout, stderr, duration_ms, now_ms()],
        )?;
        changed(n, "attempt", id)
    }

    pub fn list_attempts(&self, conversation_id: &str) -> Result<Vec<Attempt>> {
        self.all(&format!("{ATTEMPT} WHERE conversation_id = ?1 ORDER BY created_at, rowid"), [conversation_id], attempt)
    }

    // ---- settings ------------------------------------------------------------------------

    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [key, value],
        )?;
        Ok(())
    }

    // ---- maintenance ---------------------------------------------------------------------

    pub fn integrity_check(&self) -> Result<()> {
        let rows = self.all("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?;
        if rows == ["ok"] {
            Ok(())
        } else {
            Err(StoreError::Corrupt(rows.join("; ")))
        }
    }

    fn all<T>(&self, sql: &str, p: impl Params, f: fn(&Row<'_>) -> rusqlite::Result<T>) -> Result<Vec<T>> {
        let mut st = self.conn.prepare_cached(sql)?;
        let rows = st.query_map(p, f)?.collect::<rusqlite::Result<Vec<T>>>()?;
        Ok(rows)
    }
}

fn migrate(conn: &mut Connection) -> Result<()> {
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.execute_batch("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)")?;
    let current: i64 = tx.query_row("SELECT COALESCE(MAX(version), 0) FROM schema_version", [], |r| r.get(0))?;
    let current = usize::try_from(current)
        .ok()
        .filter(|&v| v <= MIGRATIONS.len())
        .ok_or_else(|| {
            StoreError::Corrupt(format!("schema version {current} unsupported (this build knows up to {})", MIGRATIONS.len()))
        })?;
    for (i, sql) in MIGRATIONS.iter().enumerate().skip(current) {
        tx.execute_batch(sql)?;
        tx.execute("INSERT INTO schema_version(version) VALUES (?1)", [i as i64 + 1])?;
    }
    tx.commit()?;
    Ok(())
}

fn changed(rows: usize, what: &'static str, id: &str) -> Result<()> {
    if rows == 0 {
        Err(StoreError::NotFound(what, id.into()))
    } else {
        Ok(())
    }
}

/// A foreign-key violation on insert means the referenced parent doesn't exist.
fn fk(e: rusqlite::Error, what: &'static str, id: &str) -> StoreError {
    match &e {
        rusqlite::Error::SqliteFailure(f, _) if f.extended_code == ffi::SQLITE_CONSTRAINT_FOREIGNKEY => {
            StoreError::NotFound(what, id.into())
        }
        _ => e.into(),
    }
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

/// 32 lowercase hex chars: wall-clock nanos, then a randomly keyed hash of a process counter
/// and those nanos (`RandomState::new()` rekeys on every call).
fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let mut h = RandomState::new().build_hasher();
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.write_u64(nanos);
    format!("{nanos:016x}{:016x}", h.finish())
}

fn decode_f32(b: &[u8]) -> Result<Vec<f32>> {
    let (words, rest) = b.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(StoreError::Corrupt(format!("embedding blob of {} bytes is not f32-aligned", b.len())));
    }
    Ok(words.iter().map(|w| f32::from_le_bytes(*w)).collect())
}

fn json_col(r: &Row<'_>, i: usize) -> rusqlite::Result<Value> {
    let s: String = r.get(i)?;
    serde_json::from_str(&s).map_err(|e| rusqlite::Error::FromSqlConversionFailure(i, Type::Text, Box::new(e)))
}

fn conversation(r: &Row<'_>) -> rusqlite::Result<Conversation> {
    Ok(Conversation {
        id: r.get(0)?,
        title: r.get(1)?,
        created_at: r.get(2)?,
        updated_at: r.get(3)?,
        mode: r.get(4)?,
        system_prompt: r.get(5)?,
        repo_path: r.get(6)?,
    })
}

fn message(r: &Row<'_>) -> rusqlite::Result<Message> {
    Ok(Message {
        id: r.get(0)?,
        conversation_id: r.get(1)?,
        seq: r.get(2)?,
        role: r.get(3)?,
        content: r.get(4)?,
        created_at: r.get(5)?,
        meta: json_col(r, 6)?,
    })
}

fn repo(r: &Row<'_>) -> rusqlite::Result<Repo> {
    Ok(Repo { id: r.get(0)?, path: r.get(1)?, added_at: r.get(2)? })
}

fn generation(r: &Row<'_>) -> rusqlite::Result<IndexGeneration> {
    Ok(IndexGeneration {
        repo_id: r.get(0)?,
        generation: r.get(1)?,
        created_at: r.get(2)?,
        finished_at: r.get(3)?,
        vector_file: r.get(4)?,
        chunk_count: r.get(5)?,
        recipe: r.get(6)?,
    })
}

fn chunk(r: &Row<'_>) -> rusqlite::Result<Chunk> {
    Ok(Chunk {
        id: r.get(0)?,
        repo_id: r.get(1)?,
        generation: r.get(2)?,
        path: r.get(3)?,
        start_line: r.get(4)?,
        end_line: r.get(5)?,
        digest: r.get(6)?,
        content: r.get(7)?,
        symbol: r.get(8)?,
        kind: r.get(9)?,
    })
}

fn attempt(r: &Row<'_>) -> rusqlite::Result<Attempt> {
    Ok(Attempt {
        id: r.get(0)?,
        conversation_id: r.get(1)?,
        message_id: r.get(2)?,
        kind: r.get(3)?,
        payload: json_col(r, 4)?,
        status: r.get(5)?,
        exit_code: r.get(6)?,
        stdout: r.get(7)?,
        stderr: r.get(8)?,
        duration_ms: r.get(9)?,
        created_at: r.get(10)?,
        updated_at: r.get(11)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Separates millisecond timestamps where a test depends on their order.
    fn tick() {
        std::thread::sleep(Duration::from_millis(2));
    }

    fn store() -> Store {
        Store::open_in_memory().unwrap()
    }

    fn new_chunk(path: &str, digest: u8) -> NewChunk {
        NewChunk {
            path: path.into(),
            start_line: 1,
            end_line: 9,
            digest: vec![digest; 32],
            content: format!("fn {path}() {{}}"),
            symbol: Some(path.into()),
            kind: Some("fn".into()),
        }
    }

    #[test]
    fn conversation_and_message_round_trip() {
        let mut s = store();
        let c = s.create_conversation("t", "chat", "be brief").unwrap();
        assert_eq!(c.id.len(), 32);
        assert!(c.id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        assert_eq!(s.get_conversation(&c.id).unwrap(), c);

        s.rename_conversation(&c.id, "renamed").unwrap();
        s.set_mode(&c.id, "code").unwrap();
        s.set_repo_path(&c.id, Some("/r")).unwrap();
        let got = s.get_conversation(&c.id).unwrap();
        assert_eq!((got.title.as_str(), got.mode.as_str(), got.repo_path.as_deref()), ("renamed", "code", Some("/r")));
        s.set_repo_path(&c.id, None).unwrap();
        assert_eq!(s.get_conversation(&c.id).unwrap().repo_path, None);

        let meta = json!({"generator": "g1", "quantization": "q4", "prompt_tokens": 3, "cancelled": false});
        let m = s.append_message(&c.id, Role::User, "hi", meta).unwrap();
        assert_eq!(s.list_messages(&c.id).unwrap(), vec![m.clone()]);
        s.update_message(&m.id, "hello", json!({"finish_reason": "stop"})).unwrap();
        let last = s.last_message(&c.id).unwrap().unwrap();
        assert_eq!((last.content.as_str(), &last.meta), ("hello", &json!({"finish_reason": "stop"})));

        assert!(matches!(s.get_conversation("nope"), Err(StoreError::NotFound("conversation", _))));
        assert!(matches!(s.update_message("nope", "", json!({})), Err(StoreError::NotFound("message", _))));
        assert!(matches!(s.rename_conversation("nope", ""), Err(StoreError::NotFound("conversation", _))));

        tick();
        let c2 = s.create_conversation("b", "chat", "").unwrap();
        tick();
        s.append_message(&c.id, Role::Assistant, "x", json!({})).unwrap();
        let ids: Vec<String> = s.list_conversations().unwrap().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, [c.id.clone(), c2.id]);
    }

    #[test]
    fn repos_settings_attempts_round_trip() {
        let s = store();
        let r = s.upsert_repo("/src/frost").unwrap();
        assert_eq!(s.upsert_repo("/src/frost").unwrap(), r);
        assert_eq!(s.list_repos().unwrap(), vec![r.clone()]);
        s.remove_repo(&r.id).unwrap();
        assert!(s.list_repos().unwrap().is_empty());
        assert!(matches!(s.remove_repo(&r.id), Err(StoreError::NotFound("repo", _))));

        assert_eq!(s.get_setting("model").unwrap(), None);
        s.set_setting("model", "a").unwrap();
        s.set_setting("model", "b").unwrap();
        assert_eq!(s.get_setting("model").unwrap().as_deref(), Some("b"));

        let c = s.create_conversation("t", "code", "").unwrap();
        let a = s.insert_attempt(&c.id, None, AttemptKind::RunCommand, json!({"cmd": "cargo test"})).unwrap();
        assert_eq!(s.list_attempts(&c.id).unwrap(), vec![a.clone()]);
        s.set_attempt_status(&a.id, AttemptStatus::Approved).unwrap();
        s.finish_attempt(&a.id, AttemptStatus::Failed, Some(101), "out", "err", Some(42)).unwrap();
        let got = &s.list_attempts(&c.id).unwrap()[0];
        assert_eq!(
            (got.status, got.exit_code, got.stdout.as_str(), got.stderr.as_str(), got.duration_ms, &got.payload),
            (AttemptStatus::Failed, Some(101), "out", "err", Some(42), &a.payload)
        );
        assert_eq!(serde_json::to_value(AttemptKind::ProposeDiff).unwrap(), "propose_diff");
        assert_eq!("timeout".parse::<AttemptStatus>().unwrap(), AttemptStatus::Timeout);
        assert!(matches!(
            s.insert_attempt("nope", None, AttemptKind::Repair, json!({})),
            Err(StoreError::NotFound("conversation", _))
        ));
    }

    #[test]
    fn delete_conversation_cascades() {
        let mut s = store();
        let c = s.create_conversation("t", "chat", "").unwrap();
        let m = s.append_message(&c.id, Role::User, "hi", json!({})).unwrap();
        s.insert_attempt(&c.id, Some(&m.id), AttemptKind::ProposeDiff, json!({"diff": ""})).unwrap();
        s.delete_conversation(&c.id).unwrap();
        let count = |t: &str| -> i64 { s.conn.query_row(&format!("SELECT COUNT(*) FROM {t}"), [], |r| r.get(0)).unwrap() };
        assert_eq!((count("conversations"), count("messages"), count("attempts")), (0, 0, 0));
    }

    #[test]
    fn seq_ordering_and_truncate_after() {
        let mut s = store();
        let c = s.create_conversation("t", "chat", "").unwrap();
        let roles = [Role::System, Role::User, Role::Assistant, Role::User, Role::Assistant];
        for (i, role) in roles.into_iter().enumerate() {
            assert_eq!(s.append_message(&c.id, role, &i.to_string(), json!({})).unwrap().seq, i as i64 + 1);
        }
        assert_eq!(s.truncate_after(&c.id, 3).unwrap(), 2);
        let seqs: Vec<i64> = s.list_messages(&c.id).unwrap().iter().map(|m| m.seq).collect();
        assert_eq!(seqs, [1, 2, 3]);
        assert_eq!(s.append_message(&c.id, Role::User, "edited", json!({})).unwrap().seq, 4);
        assert_eq!(s.last_message(&c.id).unwrap().unwrap().content, "edited");
    }

    #[test]
    fn failed_append_rolls_back() {
        let mut s = store();
        let c = s.create_conversation("t", "chat", "").unwrap();
        s.conn.execute("UPDATE conversations SET updated_at = 1 WHERE id = ?1", [&c.id]).unwrap();
        // The UPDATE of updated_at runs first inside the transaction; the CHECK then rejects the row.
        let err = s.append_raw(&c.id, "robot", "beep", json!({})).unwrap_err();
        assert!(matches!(&err, StoreError::InvalidRole(r) if r == "robot"), "{err:?}");
        assert!(s.list_messages(&c.id).unwrap().is_empty());
        assert_eq!(s.get_conversation(&c.id).unwrap().updated_at, 1);
        assert!(matches!("robot".parse::<Role>(), Err(StoreError::InvalidRole(_))));

        s.append_message(&c.id, Role::User, "hi", json!({})).unwrap();
        assert!(s.get_conversation(&c.id).unwrap().updated_at > 1);
    }

    #[test]
    fn foreign_keys_enforced() {
        let mut s = store();
        assert!(matches!(
            s.append_message("ghost", Role::User, "x", json!({})),
            Err(StoreError::NotFound("conversation", _))
        ));
        let raw = s.conn.execute(
            "INSERT INTO messages(id, conversation_id, seq, role, content, created_at) VALUES ('m', 'ghost', 1, 'user', 'x', 0)",
            [],
        );
        assert!(
            matches!(&raw, Err(rusqlite::Error::SqliteFailure(f, _)) if f.extended_code == ffi::SQLITE_CONSTRAINT_FOREIGNKEY),
            "{raw:?}"
        );
        assert!(matches!(s.begin_generation("ghost", "v", "r"), Err(StoreError::NotFound("repo", _))));
    }

    #[test]
    fn embedding_cache_lru() {
        let s = store();
        let v = [1.0f32, -2.5, 3.25, f32::MIN_POSITIVE];
        for d in [b"a", b"b", b"c"] {
            s.put_embedding("r1", d, &v).unwrap();
            tick();
        }
        assert_eq!(s.embedding_cache_bytes().unwrap(), 48);
        assert_eq!(s.get_embedding("r1", b"a").unwrap().as_deref(), Some(&v[..])); // touches "a"
        assert_eq!(s.get_embedding("r2", b"a").unwrap(), None);
        assert_eq!(s.prune_embeddings(32).unwrap(), 1); // "b" is now least recently used
        assert_eq!(s.get_embedding("r1", b"b").unwrap(), None);
        assert!(s.get_embedding("r1", b"c").unwrap().is_some());
        assert_eq!(s.embedding_cache_bytes().unwrap(), 32);
        s.put_embedding("r1", b"c", &v[..2]).unwrap(); // overwrite shrinks the entry
        assert_eq!(s.embedding_cache_bytes().unwrap(), 24);
        assert_eq!(s.prune_embeddings(0).unwrap(), 2);
        assert_eq!(s.embedding_cache_bytes().unwrap(), 0);
    }

    #[test]
    fn reopen_file_store() {
        let path = std::env::temp_dir().join(format!("frost-store-{}.db", new_id()));
        let c = {
            let mut s = Store::open(&path).unwrap();
            let c = s.create_conversation("persist", "chat", "sys").unwrap();
            s.append_message(&c.id, Role::User, "hi", json!({"k": 1})).unwrap();
            let mode: String = s.conn.query_row("PRAGMA journal_mode", [], |r| r.get(0)).unwrap();
            assert_eq!(mode, "wal");
            c
        };
        let s = Store::open(&path).unwrap(); // migrations must not re-run
        let got = s.get_conversation(&c.id).unwrap();
        assert_eq!((got.title.as_str(), got.system_prompt.as_str(), got.created_at), ("persist", "sys", c.created_at));
        assert_eq!(s.list_messages(&c.id).unwrap()[0].meta, json!({"k": 1}));
        let v: i64 = s.conn.query_row("SELECT MAX(version) FROM schema_version", [], |r| r.get(0)).unwrap();
        assert_eq!(v, MIGRATIONS.len() as i64);
        s.integrity_check().unwrap();

        s.conn.execute("INSERT INTO schema_version(version) VALUES (999)", []).unwrap();
        drop(s);
        assert!(matches!(Store::open(&path), Err(StoreError::Corrupt(_))));
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{ext}", path.display()));
        }
    }

    #[test]
    fn chunks_insert_and_ordered_lookup() {
        let mut s = store();
        let r = s.upsert_repo("/r").unwrap();
        let g = s.begin_generation(&r.id, "v1.bin", "bge-small@q8").unwrap();
        assert_eq!(g, 1);
        let ids = s.insert_chunks(&r.id, g, &[new_chunk("a", 1), new_chunk("b", 2), new_chunk("c", 1)]).unwrap();
        let paths: Vec<String> =
            s.get_chunks(&[ids[2], ids[0], 9999, ids[1]]).unwrap().into_iter().map(|c| c.path).collect();
        assert_eq!(paths, ["c", "a", "b"]);
        let all = s.chunks_for_generation(&r.id, g).unwrap();
        assert_eq!(all.iter().map(|c| c.id).collect::<Vec<_>>(), ids);
        assert_eq!(all[0].digest, vec![1; 32]);
        assert_eq!(s.find_chunks_by_digest(&r.id, &[1; 32]).unwrap().len(), 2);

        assert_eq!(s.latest_finished_generation(&r.id).unwrap(), None);
        s.finish_generation(&r.id, g, 3).unwrap();
        let fin = s.latest_finished_generation(&r.id).unwrap().unwrap();
        assert_eq!((fin.generation, fin.chunk_count, fin.vector_file.as_str()), (1, 3, "v1.bin"));
        assert!(fin.finished_at.is_some());
        assert!(matches!(
            s.insert_chunks(&r.id, 99, &[new_chunk("x", 3)]),
            Err(StoreError::NotFound("generation", _))
        ));
    }

    #[test]
    fn prune_generations_keeps_newest() {
        let mut s = store();
        let r = s.upsert_repo("/r").unwrap();
        for g in 1..=4 {
            assert_eq!(s.begin_generation(&r.id, &format!("v{g}"), "x").unwrap(), g);
            s.insert_chunks(&r.id, g, &[new_chunk("a", g as u8)]).unwrap();
            s.finish_generation(&r.id, g, 1).unwrap();
        }
        let building = s.begin_generation(&r.id, "v5", "x").unwrap();
        assert_eq!(s.prune_generations(&r.id, 2).unwrap(), 2);
        let left: Vec<i64> =
            s.all("SELECT generation FROM index_generations ORDER BY generation", [], |r| r.get(0)).unwrap();
        assert_eq!(left, [3, 4, building]); // in-progress build survives
        assert!(s.chunks_for_generation(&r.id, 1).unwrap().is_empty()); // chunks cascaded
        assert_eq!(s.chunks_for_generation(&r.id, 4).unwrap().len(), 1);
        s.integrity_check().unwrap();
    }
}

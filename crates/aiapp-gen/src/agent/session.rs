//! Sessions: one user conversation == one independent session.
//!
//! A session owns its own conversation history (rendered layer per session),
//! persists to `config.sessions_dir/<id>.json`, and is fully isolated from
//! other sessions. A user may hold many concurrent sessions.

use crate::agent::types::Message;
use crate::GenError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionFile {
    id: String,
    scope: String,
    messages: Vec<Message>,
}

/// A single independent conversation.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub scope: String,
    messages: Vec<Message>,
    /// Where this session is persisted (empty = memory-only).
    file: PathBuf,
}

impl Session {
    /// Create a new session for `scope` (a user id). Persists to `sessions_dir`.
    pub fn new(scope: &str, sessions_dir: &std::path::Path) -> Result<Self, GenError> {
        let id = format!(
            "s-{}-{}",
            chrono::Local::now().format("%Y%m%d%H%M%S"),
            session_rand_suffix()
        );
        let file = sessions_dir.join(format!("{id}.json"));
        let mut s = Self {
            id,
            scope: scope.to_string(),
            messages: Vec::new(),
            file,
        };
        s.persist()?;
        Ok(s)
    }

    /// Open an existing session by id.
    pub fn open(id: &str, sessions_dir: &std::path::Path) -> Result<Self, GenError> {
        let file = sessions_dir.join(format!("{id}.json"));
        let raw = std::fs::read_to_string(&file).map_err(GenError::Io)?;
        let sf: SessionFile = serde_json::from_str(&raw)
            .map_err(|e| GenError::Backend("session", format!("parse {file:?}: {e}")))?;
        Ok(Self {
            id: sf.id,
            scope: sf.scope,
            messages: sf.messages,
            file,
        })
    }

    /// Read the raw conversation history.
    pub fn history(&self) -> &[Message] {
        &self.messages
    }

    /// Append a message and persist.
    pub fn push(&mut self, m: Message) -> Result<(), GenError> {
        self.messages.push(m);
        self.persist()
    }

    /// Persist the current history.
    pub fn persist(&self) -> Result<(), GenError> {
        if let Some(parent) = self.file.parent() {
            std::fs::create_dir_all(parent).map_err(GenError::Io)?;
        }
        let sf = SessionFile {
            id: self.id.clone(),
            scope: self.scope.clone(),
            messages: self.messages.clone(),
        };
        let j = serde_json::to_string_pretty(&sf)
            .map_err(|e| GenError::Backend("session", e.to_string()))?;
        std::fs::write(&self.file, j).map_err(GenError::Io)?;
        Ok(())
    }

    /// Last N messages (for context windows).
    pub fn recent(&self, n: usize) -> Vec<Message> {
        let start = self.messages.len().saturating_sub(n);
        self.messages[start..].to_vec()
    }
}

/// Short random suffix to avoid id collisions.
fn session_rand_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or_default();
    format!("{:06x}", nanos)
}
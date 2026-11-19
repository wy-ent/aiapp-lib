//! Cross-session long-term memory. Persisted to a JSON file so that a user's
//! preferences and context survive across independent conversations/sessions.
//!
//! Memory is keyed by a `scope` (typically a user id). Agents can `remember` a
//! fact and `recall` it in a later session — the foundation of "remember me".

use crate::GenError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MemoryFile {
    /// scope_id -> key -> value.
    scopes: HashMap<String, HashMap<String, serde_json::Value>>,
}

/// In-memory + file-backed store. Callers usually build it once per engine and
/// share it (with interior mutability) across sessions.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    file: std::path::PathBuf,
    #[allow(clippy::mutable_key_type)]
    data: std::sync::Arc<std::sync::Mutex<MemoryFile>>,
}

impl MemoryStore {
    /// Open (or lazily create) the store at `path`.
    pub fn open(path: &Path) -> Result<Self, GenError> {
        let data = if path.exists() {
            let raw = std::fs::read_to_string(path).map_err(GenError::Io)?;
            serde_json::from_str(&raw).map_err(|e| {
                GenError::Backend("memory", format!("parse {}: {e}", path.display()))
            })?
        } else {
            MemoryFile::default()
        };
        Ok(Self {
            file: path.to_path_buf(),
            data: std::sync::Arc::new(std::sync::Mutex::new(data)),
        })
    }

    /// Persist current state to disk.
    pub fn persist(&self) -> Result<(), GenError> {
        let guard = self.data.lock().map_err(|_| {
            GenError::Backend("memory", "lock poisoned".into())
        })?;
        if let Some(parent) = self.file.parent() {
            std::fs::create_dir_all(parent).map_err(GenError::Io)?;
        }
        let j = serde_json::to_string_pretty(&*guard)
            .map_err(|e| GenError::Backend("memory", format!("serialize: {e}")))?;
        std::fs::write(&self.file, j).map_err(GenError::Io)?;
        Ok(())
    }

    /// Remember a `key = value` for a `scope`. Persists immediately.
    pub fn remember(
        &self,
        scope: &str,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), GenError> {
        self.data
            .lock()
            .map_err(|_| GenError::Backend("memory", "lock poisoned".into()))?
            .scopes
            .entry(scope.to_string())
            .or_default()
            .insert(key.to_string(), value);
        self.persist()
    }

    /// Recall a remembered value for `scope`.
    pub fn recall(&self, scope: &str, key: &str) -> Option<serde_json::Value> {
        self.data
            .lock()
            .ok()?
            .scopes
            .get(scope)?
            .get(key)
            .cloned()
    }

    /// Forget a key for `scope`.
    pub fn forget(&self, scope: &str, key: &str) -> Result<(), GenError> {
        if let Some(map) = self
            .data
            .lock()
            .map_err(|_| GenError::Backend("memory", "lock poisoned".into()))?
            .scopes
            .get_mut(scope)
        {
            map.remove(key);
        }
        self.persist()
    }

    /// Dump all remembered key/value pairs for a `scope` as free text (used to
    /// build a "what I know about you" context block).
    pub fn dump(&self, scope: &str) -> String {
        let guard = match self.data.lock() {
            Ok(g) => g,
            Err(_) => return String::new(),
        };
        let Some(map) = guard.scopes.get(scope) else {
            return String::new();
        };
        if map.is_empty() {
            return String::new();
        }
        let mut lines: Vec<String> = map
            .iter()
            .map(|(k, v)| format!("- {k}: {v}"))
            .collect();
        lines.sort();
        lines.join("\n")
    }
}
//! Tool system: capabilities an agent can invoke inside its run loop.
//!
//! Each tool is a plain function `(ToolCtx, args: Value) -> String`. Agents
//! reference tools via a lightweight text protocol (see `modes.rs::ToolLoop`):
//!
//! ```text
//! ```tool
//! {"name":"file_write","args":{"path":"a.txt","content":"hi"}}
//! ```
//! ```
//!
//! Fencing: all `file_*` tools resolve paths against `config.root_dir`; any
//! path attempting to escape the root is rejected.

pub mod fileops;
pub mod runner;
pub mod websearch;

use crate::agent::config::AgentConfig;
use crate::agent::memory::MemoryStore;
use crate::agent::registry::Registry;
use serde::Deserialize;
use serde_json::Value;

/// Shared context handed to every tool call.
#[derive(Clone)]
pub struct ToolCtx {
    pub config: AgentConfig,
    pub memory: MemoryStore,
    pub registry: Registry,
    /// Scope id used by memory tools (typically the user id).
    pub scope: String,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Map<String, Value>,
}

/// Resolve a (possibly relative) path inside the fenced root. Returns `None` if
/// the path escapes the root.
pub fn resolve_fenced(root: &std::path::Path, p: &str) -> Option<std::path::PathBuf> {
    let base = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let joined = root.join(p);
    let canon = joined.canonicalize().unwrap_or(joined.clone());
    if canon.starts_with(&base) {
        Some(canon)
    } else {
        None
    }
}

/// Dispatch a tool by name + JSON args. Returns the text result (errors embedded).
pub async fn run_tool(ctx: &ToolCtx, name: &str, args: Value) -> String {
    let folded: Result<String, crate::GenError> = match name {
        "file_read" => fileops::read(ctx, args),
        "file_write" => fileops::write(ctx, args),
        "file_append" => fileops::append(ctx, args),
        "file_list" => fileops::list(ctx, args),
        "file_search" => fileops::search(ctx, args),
        "web_search" => websearch::search(ctx, args).await,
        "fetch_url" => websearch::fetch(ctx, args).await,
        "memory_recall" => {
            let scope = &ctx.scope;
            let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("");
            match ctx.memory.recall(scope, key) {
                Some(v) => Ok(format!("MEMORY {key}: {v}")),
                None => Ok(format!("MEMORY {key}: (not found)")),
            }
        }
        "memory_remember" => {
            let scope = &ctx.scope;
            let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("");
            let val = args.get("value").cloned().unwrap_or(Value::Null);
            ctx.memory
                .remember(scope, key, val)
                .map(|_| format!("remembered {key}"))
        }
        "memory_dump" => {
            let dump = ctx.memory.dump(&ctx.scope);
            if dump.is_empty() {
                Ok("no remembered facts".to_string())
            } else {
                Ok(dump)
            }
        }
        "run_cmd" => runner::run_cmd(ctx, args).await,
        "start_preview" => runner::open_preview(ctx, args).await,
        other => Ok(format!("UNKNOWN TOOL: {other}")),
    };
    match folded {
        Ok(s) => s,
        Err(e) => format!("TOOL ERROR: {e}"),
    }
}

/// Parse a tool-call block from generated text. Returns the first tool call found.
pub fn parse_tool_block(text: &str) -> Option<ToolCall> {
    // Look for a ```tool / ```json fence and try to parse its inner JSON.
    for marker in ["```tool", "```json", "```TOOL"] {
        if let Some(start) = text.find(marker) {
            let rest = &text[start + marker.len()..];
            if let Some(end) = rest.find("```") {
                let inner = rest[..end].trim();
                if let Ok(call) = serde_json::from_str::<ToolCall>(inner) {
                    return Some(call);
                }
            }
        }
    }
    // No fence matched: try the whole trimmed text as a single tool-call object.
    serde_json::from_str::<ToolCall>(text.trim()).ok()
}

/// Render a tool result into a `Message` of role `Tool`.
pub fn tool_message(name: &str, result: String) -> crate::agent::types::Message {
    crate::agent::types::Message {
        role: crate::agent::types::MsgRole::Tool,
        name: name.to_string(),
        content: result,
        meta: None,
    }
}
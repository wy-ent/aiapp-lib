//! Project runner + preview tooling. Lets an agent start a frontend/backend
//! dev server inside the fenced root and open a browser preview — the final
//! "idea → running project → preview" step of a workflow.

use super::resolve_fenced;
use super::ToolCtx;
use crate::GenError;
use serde_json::{Map, Value};
use std::process::Stdio;
use std::time::Duration;

#[inline]
fn str_arg(args: &Map<String, Value>, key: &str) -> Result<String, GenError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| GenError::Backend("runner", format!("missing arg `{key}`")))
}

/// Run a shell command, confined to the fenced root. `{cmd, cwd?, background?, timeout?}`.
pub async fn run_cmd(ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let args_obj = args.as_object().cloned().unwrap_or_default();
    let a = &args_obj;
    let cmd = str_arg(a, "cmd")?;
    let cwd = str_arg(a, "cwd").unwrap_or_else(|_| ".".into());
    let background = a.get("background").and_then(|v| v.as_bool()).unwrap_or(false);
    let timeout = a.get("timeout").and_then(|v| v.as_u64()).unwrap_or(30);

    let cwd_abs = if cwd.is_empty() {
        ctx.config.root_dir.clone()
    } else {
        resolve_fenced(&ctx.config.root_dir, &cwd)
            .ok_or_else(|| GenError::Backend("runner", format!("FENCE: cwd `{cwd}` escapes root")))?
    };

    let mut parts = shell_words(&cmd);
    if parts.is_empty() {
        return Err(GenError::Backend("runner", "empty command".into()));
    }
    let program = parts.remove(0);

    if background {
        // Detached background server.
        let child = tokio::process::Command::new(&program)
            .args(&parts)
            .current_dir(&cwd_abs)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| GenError::Backend("runner", format!("spawn {program}: {e}")))?;
        return Ok(format!("started background `{}` (pid {}) in {}", program, child.id().unwrap_or_default(), cwd_abs.display()));
    }

    let mut child = tokio::process::Command::new(&program)
        .args(&parts)
        .current_dir(&cwd_abs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| GenError::Backend("runner", format!("spawn {program}: {e}")))?;

    let output = match tokio::time::timeout(
        Duration::from_secs(timeout),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Err(GenError::Backend("runner", format!("wait: {e}"))),
        Err(_) => return Err(GenError::Backend("runner", format!("timed out after {timeout}s"))),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Ok(format!(
        "exit={}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status.code().unwrap_or(-1),
        truncate(&stdout, 800),
        truncate(&stderr, 800)
    ))
}

/// Start a preview for a project and open it in the browser.
/// `{cwd?, command?, port?, auto_open?}` — if `command` is omitted, serves the
/// directory with a static server.
pub async fn open_preview(ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let args_obj = args.as_object().cloned().unwrap_or_default();
    let a = &args_obj;
    let cwd = str_arg(a, "cwd").unwrap_or_else(|_| ".".into());
    let command = str_arg(a, "command").unwrap_or_default();
    let port = a.get("port").and_then(|v| v.as_u64()).unwrap_or(8787);
    let auto_open = a.get("auto_open").and_then(|v| v.as_bool()).unwrap_or(true);

    let cwd_abs = if cwd.is_empty() {
        ctx.config.root_dir.clone()
    } else {
        resolve_fenced(&ctx.config.root_dir, &cwd)
            .ok_or_else(|| GenError::Backend("runner", format!("FENCE: cwd `{cwd}` escapes root")))?
    };
    let rel = relative_to(&cwd_abs, &ctx.config.root_dir);

    // Determine the start command: explicit command, else a static server.
    if command.is_empty() {
        run_cmd(
            ctx,
            serde_json::json!({ "cmd": format!("python3 -m http.server {port}"), "cwd": rel, "background": true }),
        )
        .await?;
    } else {
        run_cmd(ctx, serde_json::json!({ "cmd": command, "cwd": rel, "background": true })).await?;
    }

    let url = if rel.is_empty() {
        format!("http://127.0.0.1:{port}/")
    } else {
        format!("http://127.0.0.1:{port}/{}", rel)
    };

    if auto_open {
        // Best-effort: open the system browser.
        if std::process::Command::new("open")
            .arg(&url)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return Ok(format!("preview opened in browser: {url}"));
        }
    }
    Ok(format!("preview started: {url}"))
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}\n... (truncated)")
    }
}

/// Compute a POSIX-style relative path from `abs` to `base`. Returns `".."`-free
/// relative string if `abs` is under `base`, else an empty string.
fn relative_to(abs: &std::path::Path, base: &std::path::Path) -> String {
    let abs = abs.canonicalize().unwrap_or_else(|_| abs.to_path_buf());
    let base = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());
    match abs.strip_prefix(&base) {
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => String::new(),
    }
}

/// Minimal shell word splitter (handles spaces and basic quotes).
fn shell_words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                } else {
                    cur.push(c);
                }
            }
            None => match c {
                '\'' | '"' => quote = Some(c),
                ' ' | '\t' => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                }
                _ => cur.push(c),
            },
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
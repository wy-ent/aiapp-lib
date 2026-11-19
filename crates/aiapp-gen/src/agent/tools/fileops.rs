//! Local file tools, confined to the fenced root directory. Any path escaping
//! the root is rejected (`FENCE`), so agents can never touch files outside the
//! sandbox.

use super::resolve_fenced;
use super::ToolCtx;
use crate::GenError;
use serde_json::{Map, Value};

#[inline]
fn str_arg(args: &Map<String, Value>, key: &str) -> Result<String, GenError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| GenError::Backend("file_ops", format!("missing arg `{key}`")))
}

/// Read a file (text). `{path}`.
pub fn read(ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let path = str_arg(args.as_object().unwrap_or(&Map::new()), "path")?;
    let p = resolve_fenced(&ctx.config.root_dir, &path)
        .ok_or_else(|| GenError::Backend("file_ops", format!("FENCE: {path} escapes root")))?;
    std::fs::read_to_string(&p).map_err(GenError::Io)
}

/// Write a file (creating parent dirs). `{path, content}`.
pub fn write(ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let args_obj = args.as_object().cloned().unwrap_or_default();
    let a = &args_obj;
    let path = str_arg(a, "path")?;
    let content = str_arg(a, "content").unwrap_or_default();
    let p = resolve_fenced(&ctx.config.root_dir, &path)
        .ok_or_else(|| GenError::Backend("file_ops", format!("FENCE: {path} escapes root")))?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(GenError::Io)?;
    }
    std::fs::write(&p, content).map_err(GenError::Io)?;
    Ok(format!("wrote {}", p.display()))
}

/// Append to a file. `{path, content}`.
pub fn append(ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let args_obj = args.as_object().cloned().unwrap_or_default();
    let a = &args_obj;
    let path = str_arg(a, "path")?;
    let content = str_arg(a, "content").unwrap_or_default();
    let p = resolve_fenced(&ctx.config.root_dir, &path)
        .ok_or_else(|| GenError::Backend("file_ops", format!("FENCE: {path} escapes root")))?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(GenError::Io)?;
    }
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)
        .map_err(GenError::Io)?;
    f.write_all(content.as_bytes()).map_err(GenError::Io)?;
    Ok(format!("appended {}", p.display()))
}

/// List a directory. `{path, recursive?}`.
pub fn list(ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let args_obj = args.as_object().cloned().unwrap_or_default();
    let a = &args_obj;
    let path = str_arg(a, "path").unwrap_or_else(|_| ".".into());
    let recursive = a.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false);
    let p = resolve_fenced(&ctx.config.root_dir, &path)
        .ok_or_else(|| GenError::Backend("file_ops", format!("FENCE: {path} escapes root")))?;
    let mut out = Vec::new();
    if recursive {
        walk(&p, &mut out, 0, 12)?;
    } else {
        for e in std::fs::read_dir(&p).map_err(GenError::Io)? {
            let e = e.map_err(GenError::Io)?;
            out.push(e.file_name().to_string_lossy().into_owned());
        }
    }
    out.sort();
    Ok(out.join("\n"))
}

fn walk(dir: &std::path::Path, out: &mut Vec<String>, depth: usize, max: usize) -> Result<(), GenError> {
    if depth > max {
        return Ok(());
    }
    for e in std::fs::read_dir(dir).map_err(GenError::Io)? {
        let e = e.map_err(GenError::Io)?;
        let name = e.file_name().to_string_lossy().into_owned();
        if e.file_type().map_err(GenError::Io)?.is_dir() {
            out.push(format!("{}{}/", "  ".repeat(depth), name));
            walk(&e.path(), out, depth + 1, max)?;
        } else {
            out.push(format!("{}{}", "  ".repeat(depth), name));
        }
    }
    Ok(())
}

/// Search files by name glob inside the root. `{pattern}` e.g. `**/*.rs`.
pub fn search(ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let args_obj = args.as_object().cloned().unwrap_or_default();
    let a = &args_obj;
    let pattern = str_arg(a, "pattern").unwrap_or_else(|_| "*".into());
    let root = ctx.config.root_dir.clone();
    let joined = root.join(&pattern);
    // Guard: do not let the glob base escape the root.
    let base = root.canonicalize().unwrap_or(root.clone());
    let matches: Vec<String> = glob::glob(&joined.to_string_lossy())
        .map_err(|e| GenError::Backend("file_ops", e.to_string()))?
        .filter_map(|r| r.ok())
        .filter(|p| p.canonicalize().map(|c| c.starts_with(&base)).unwrap_or(false))
        .map(|p| p.strip_prefix(&root).unwrap_or(&p).to_string_lossy().into_owned())
        .collect();
    if matches.is_empty() {
        Ok(format!("(no files matching `{pattern}`)"))
    } else {
        Ok(matches.join("\n"))
    }
}
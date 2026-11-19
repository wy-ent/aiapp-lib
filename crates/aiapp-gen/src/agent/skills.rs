//! Skill packages: import reusable knowledge packs shipped as `.zip` archives
//! (with scripts and reference material). Importing registers the skill in the
//! registry and installs its files into `config.skills_dir`.
//!
//! A skill package is a zip whose root contains a manifest:
//! - `SKILL.md` front-matter: `name`, `version`, `description`, `instruction`,
//!   and optional `scripts` / `references` / `needs_tools` lists.
//! - or a `skill.json` with the same fields.

use crate::agent::registry::Registry;
use crate::agent::types::Skill;
use crate::GenError;
use serde::Deserialize;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

/// Number of bytes to allow reading a single `SKILL.md` / manifest.
const MAX_MANIFEST_BYTES: u64 = 256 * 1024;

/// Unicode-aware, filesystem-safe slug for a skill name (keeps CJK characters).
fn slugify_skill(name: &str) -> String {
    let mut out = String::new();
    let mut last_underscore = false;
    for c in name.trim().to_lowercase().chars() {
        if c.is_alphanumeric() {
            out.push(c);
            last_underscore = false;
        } else if !last_underscore && !out.is_empty() {
            out.push('_');
            last_underscore = true;
        }
    }
    if out.is_empty() {
        "skill".to_string()
    } else {
        out
    }
}

#[derive(Debug, Default, Deserialize)]
struct SkillManifest {
    #[serde(default)]
    name: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    instruction: String,
    #[serde(default)]
    scripts: Vec<String>,
    #[serde(default)]
    references: Vec<String>,
    #[serde(default)]
    needs_tools: Vec<String>,
}

/// Import a `.zip` skill package. Returns the installed `Skill`.
///
/// The skill id is derived from the manifest `name` (slugified); the zip is
/// extracted into `skills_dir/<id>/`.
pub fn import_skill_zip(
    archive_path: &Path,
    config: &crate::agent::config::AgentConfig,
    registry: &Registry,
) -> Result<Skill, GenError> {
    if !archive_path.exists() {
        return Err(GenError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("skill archive not found: {}", archive_path.display()),
        )));
    }
    let file = std::fs::File::open(archive_path).map_err(GenError::Io)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| GenError::Backend("skills", format!("open zip: {e}")))?;

    // 1. Read manifest first (SKILL.md front-matter or skill.json).
    let manifest = read_manifest(&mut archive).unwrap_or_default();
    let name = if manifest.name.is_empty() {
        archive_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "skill".to_string())
    } else {
        manifest.name.clone()
    };
    let id = slugify_skill(&name);
    let dest = config.skills_dir.join(&id);
    if dest.exists() {
        std::fs::remove_dir_all(&dest).map_err(GenError::Io)?;
    }
    std::fs::create_dir_all(&dest).map_err(GenError::Io)?;

    // 2. Extract all entries, confining to `dest` (defends against zip-slip).
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| GenError::Backend("skills", e.to_string()))?;
        let raw_name = entry.name().to_string();
        let cleaned = raw_name.trim_start_matches('/').to_string();
        let path = dest.join(&cleaned);
        if !path.starts_with(&dest) {
            continue; // skip path that would escape the skill dir
        }
        if entry.is_dir() {
            std::fs::create_dir_all(&path).map_err(GenError::Io)?;
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(GenError::Io)?;
            }
            let mut out = std::fs::File::create(&path).map_err(GenError::Io)?;
            std::io::copy(&mut entry, &mut out).map_err(GenError::Io)?;
        }
    }

    let skill = Skill {
        id: id.clone(),
        name,
        version: manifest.version,
        description: manifest.description,
        instruction: manifest.instruction,
        scripts: manifest.scripts,
        references: manifest.references,
        needs_tools: manifest.needs_tools,
    };
    registry.create_skill(skill.clone())?;
    Ok(skill)
}

/// Read the manifest (`skill.json` first, then `SKILL.md` front-matter).
fn read_manifest(archive: &mut ZipArchive<std::fs::File>) -> Result<SkillManifest, GenError> {
    let mut found: Option<String> = None;
    for i in 0..archive.len() {
        let name = {
            let e = archive.by_index(i).map_err(|err| GenError::Backend("skills", err.to_string()))?;
            e.name().to_string()
        };
        let base = name.trim_start_matches('/');
        if base == "skill.json" || base.ends_with("/skill.json") {
            found = Some(name);
            break;
        }
        if base == "SKILL.md" || base.ends_with("/SKILL.md") {
            found = Some(name);
            break;
        }
    }
    let Some(name) = found else {
        return Ok(SkillManifest::default());
    };
    let mut entry = archive
        .by_name(&name)
        .map_err(|e| GenError::Backend("skills", e.to_string()))?;
    if name.ends_with(".json") {
        let mut buf = String::new();
        let mut limited = entry.take(MAX_MANIFEST_BYTES);
        limited.read_to_string(&mut buf).map_err(GenError::Io)?;
        serde_json::from_str(&buf)
            .map_err(|e| GenError::Backend("skills", format!("parse {name}: {e}")))
    } else {
        // Parse SKILL.md front-matter: `---\nkey: value\n---`.
        let mut buf = String::new();
        let mut limited = entry.take(MAX_MANIFEST_BYTES);
        limited.read_to_string(&mut buf).map_err(GenError::Io)?;
        Ok(parse_frontmatter(&buf))
    }
}

/// Parse YAML-ish front-matter from `SKILL.md`. Supports keys and simple lists.
fn parse_frontmatter(body: &str) -> SkillManifest {
    let mut m = SkillManifest::default();
    let Some(start) = body.find("---") else {
        // No front-matter; use the first heading line as name.
        if let Some(first) = body.lines().find(|l| l.starts_with('#')) {
            m.name = first.trim_start_matches('#').trim().to_string();
        }
        return m;
    };
    let after = &body[start + 3..];
    let Some(rel_end) = after.find("---") else {
        return m;
    };
    let block = &after[..rel_end];
    let mut current_list: Option<&'static str> = None;
    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim();
            let v = v.trim().trim_matches('"').to_string();
            current_list = match k {
                "name" => {
                    m.name = v;
                    None
                }
                "version" => {
                    m.version = v;
                    None
                }
                "description" => {
                    m.description = v;
                    None
                }
                "instruction" => {
                    m.instruction = v.replace("\\n", "\n");
                    None
                }
                "scripts" => Some("scripts"),
                "references" => Some("references"),
                "needs_tools" => Some("needs_tools"),
                _ => None,
            };
        } else if let Some(list) = current_list {
            let item = line.trim_start_matches('-').trim().trim_matches('"').to_string();
            if !item.is_empty() {
                match list {
                    "scripts" => m.scripts.push(item),
                    "references" => m.references.push(item),
                    "needs_tools" => m.needs_tools.push(item),
                    _ => {}
                }
            }
        }
    }
    m
}
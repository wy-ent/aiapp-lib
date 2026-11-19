//! Asset registry: the shared, persistent home for capabilities, roles, skills
//! and personas. ``Asset creation`` happens here: besides the built-in defaults,
//! new roles/skills/capabilities/persona can be created programmatically — the
//! "asset craftsman" agent parses natural language and calls these functions.

use crate::agent::types::{Capability, Role, Skill};
use crate::GenError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RegistryData {
    capabilities: HashMap<String, Capability>,
    roles: HashMap<String, Role>,
    skills: HashMap<String, Skill>,
}

/// Built-in role ids.
pub mod roles {
    /// Director: plans a workflow, picks the orchestration mode and assembles the team.
    pub const DIRECTOR: &str = "director";
    /// Primary assistant: provides answers/analysis directly (single-agent mode).
    pub const ASSISTANT: &str = "assistant";
    /// Implementer worker: produces concrete output (code, prose, artifacts).
    pub const WORKER: &str = "worker";
    /// Reviewer: critiques and gives feedback (used in debate).
    pub const REVIEWER: &str = "reviewer";
    /// Tool runner: responsible for tool-call loops.
    pub const OPERATOR: &str = "operator";
}

/// Built-in capability ids.
pub mod caps {
    pub const CONVERSATION: &str = "conversation";
    pub const CODING: &str = "coding";
    pub const FILE_OPS: &str = "file_ops";
    pub const WEB_SEARCH: &str = "web_search";
    pub const MEMORY: &str = "memory";
    pub const RESEARCH: &str = "research";
}

/// Persistent registry with built-in defaults.
#[derive(Debug, Clone)]
pub struct Registry {
    file: Option<std::path::PathBuf>,
    data: std::sync::Arc<std::sync::Mutex<RegistryData>>,
}

fn builtin_capabilities() -> Vec<Capability> {
    vec![
        Capability {
            id: caps::CONVERSATION.into(),
            name: "conversation".into(),
            description: "Answer questions, provide analysis and advice in direct dialogue.".into(),
        },
        Capability {
            id: caps::CODING.into(),
            name: "coding".into(),
            description: "Write and build runnable applications (MoonBit/Web).".into(),
        },
        Capability {
            id: caps::FILE_OPS.into(),
            name: "file_ops".into(),
            description: "Read, write and search files inside the fenced workspace.".into(),
        },
        Capability {
            id: caps::WEB_SEARCH.into(),
            name: "web_search".into(),
            description: "Search the web in real time and read page content.".into(),
        },
        Capability {
            id: caps::MEMORY.into(),
            name: "memory".into(),
            description: "Persist and recall cross-session user context and facts.".into(),
        },
        Capability {
            id: caps::RESEARCH.into(),
            name: "research".into(),
            description: "Gather, filter and summarise information from many sources.".into(),
        },
    ]
}

fn builtin_roles() -> Vec<Role> {
    vec![
        Role {
            id: roles::DIRECTOR.into(),
            name: "导演 (Director)".into(),
            persona: "你是编排导演。你把用户意图拆解为协作工作流：选择编排模式（顺序/并行/主管-工人/辩论/群聊/工具链），挑选团队成员（角色）与技能，分配子任务。".into(),
            capabilities: vec![
                caps::CONVERSATION.into(),
                caps::RESEARCH.into(),
            ],
            skills: vec![],
            prefer_mode: None,
        },
        Role {
            id: roles::ASSISTANT.into(),
            name: "主助手 (Assistant)".into(),
            persona: "你是主助手，直接回答用户问题、提供分析与建议。当用户只是提问时不组建团队；当用户要“做成一件事”时，调用编排引擎组建多 Agent 团队。".into(),
            capabilities: vec![caps::CONVERSATION.into(), caps::RESEARCH.into()],
            skills: vec![],
            prefer_mode: None,
        },
        Role {
            id: roles::WORKER.into(),
            name: "执行者 (Worker)".into(),
            persona: "你是执行者。专注产出可交付结果：代码、方案、文案或结构化产物，直接给出完整、自洽的输出，不闲聊。".into(),
            capabilities: vec![
                caps::CODING.into(),
                caps::FILE_OPS.into(),
                caps::WEB_SEARCH.into(),
            ],
            skills: vec![],
            prefer_mode: None,
        },
        Role {
            id: roles::REVIEWER.into(),
            name: "评审 (Reviewer)".into(),
            persona: "你是严格评审。指出问题、提出改进建议、核查完整性、一致性与合规性。给出结构化反馈清单。".into(),
            capabilities: vec![caps::CONVERSATION.into(), caps::RESEARCH.into()],
            skills: vec![],
            prefer_mode: None,
        },
        Role {
            id: roles::OPERATOR.into(),
            name: "操作者 (Operator)".into(),
            persona: "你是工具操作者。通过工具调用循环一步步完成任务：写文件、读文件、搜索、运行，并汇报结果。".into(),
            capabilities: vec![caps::FILE_OPS.into(), caps::WEB_SEARCH.into(), caps::MEMORY.into()],
            skills: vec![],
            prefer_mode: None,
        },
    ]
}

impl Registry {
    /// Open a persistent registry (loads an existing file if present), else create with defaults.
    pub fn open(path: Option<&Path>) -> Result<Self, GenError> {
        let data = match path {
            Some(p) if p.exists() => {
                let raw = std::fs::read_to_string(p).map_err(GenError::Io)?;
                serde_json::from_str(&raw)
                    .map_err(|e| GenError::Backend("registry", format!("parse: {e}")))?
            }
            _ => RegistryData::default(),
        };
        let reg = Self {
            file: path.map(Path::to_path_buf),
            data: std::sync::Arc::new(std::sync::Mutex::new(data)),
        };
        // Seed defaults if empty.
        {
            let mut d = reg.data.lock().map_err(|_| GenError::Backend("registry", "lock".into()))?;
            if d.capabilities.is_empty() {
                for c in builtin_capabilities() {
                    d.capabilities.insert(c.id.clone(), c);
                }
                for r in builtin_roles() {
                    d.roles.insert(r.id.clone(), r);
                }
            }
        }
        reg.persist()?;
        Ok(reg)
    }

    /// In-memory registry (no persistence) with defaults.
    pub fn in_memory() -> Result<Self, GenError> {
        Self::open(None)
    }

    fn persist(&self) -> Result<(), GenError> {
        if let Some(p) = &self.file {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).map_err(GenError::Io)?;
            }
            let d = self.data.lock().map_err(|_| GenError::Backend("registry", "lock".into()))?;
            let j = serde_json::to_string_pretty(&*d).map_err(|e| GenError::Backend("registry", e.to_string()))?;
            std::fs::write(p, j).map_err(GenError::Io)?;
        }
        Ok(())
    }

    // --- Capabilities ---
    pub fn capabilities(&self) -> Vec<Capability> {
        let d = self.data.lock().unwrap();
        d.capabilities.values().cloned().collect()
    }

    pub fn capability(&self, id: &str) -> Option<Capability> {
        self.data.lock().unwrap().capabilities.get(id).cloned()
    }

    pub fn create_capability(&self, c: Capability) -> Result<(), GenError> {
        self.data
            .lock()
            .map_err(|_| GenError::Backend("registry", "lock".into()))?
            .capabilities
            .insert(c.id.clone(), c);
        self.persist()
    }

    // --- Roles / personas ---
    pub fn roles(&self) -> Vec<Role> {
        let d = self.data.lock().unwrap();
        d.roles.values().cloned().collect()
    }

    pub fn role(&self, id: &str) -> Option<Role> {
        self.data.lock().unwrap().roles.get(id).cloned()
    }

    /// Create a role (asset creation). Capacity + skills get validated against existing assets.
    pub fn create_role(&self, mut r: Role) -> Result<(), GenError> {
        let mut d = self.data.lock().map_err(|_| GenError::Backend("registry", "lock".into()))?;
        r.capabilities.retain(|c| d.capabilities.contains_key(c));
        r.skills.retain(|s| d.skills.contains_key(s));
        d.roles.insert(r.id.clone(), r);
        drop(d);
        self.persist()
    }

    pub fn delete_role(&self, id: &str) -> Result<(), GenError> {
        self.data.lock().map_err(|_| GenError::Backend("registry", "lock".into()))?.roles.remove(id);
        self.persist()
    }

    // --- Skills ---
    pub fn skills(&self) -> Vec<Skill> {
        let d = self.data.lock().unwrap();
        d.skills.values().cloned().collect()
    }

    pub fn skill(&self, id: &str) -> Option<Skill> {
        self.data.lock().unwrap().skills.get(id).cloned()
    }

    pub fn create_skill(&self, s: Skill) -> Result<(), GenError> {
        self.data
            .lock()
            .map_err(|_| GenError::Backend("registry", "lock".into()))?
            .skills
            .insert(s.id.clone(), s);
        self.persist()
    }

    pub fn delete_skill(&self, id: &str) -> Result<(), GenError> {
        self.data.lock().map_err(|_| GenError::Backend("registry", "lock".into()))?.skills.remove(id);
        self.persist()
    }
}
//! Multi-agent orchestration engine for aiapp-gen.
//!
//! The vision: **one user conversation == one independent session**; the core is
//! "idea → execution" — when a single agent cannot do it, a *team* of agents
//! does. The engine supports:
//!
//! - **Direct conversation** — the primary assistant answers, analyses, advises.
//! - **Team orchestration** — dynamically assemble a multi-agent team and run a
//!   collaborative workflow across **6 orchestration modes**.
//! - **Asset creation** — create roles, capabilities, skills and persona through
//!   natural conversation.
//! - **Long-term memory** — remember preferences/context across sessions.
//! - **File operations** — read/write/search local files inside a fenced dir.
//! - **Web search** — real-time web search + page content reading.
//! - **Skill packages** — import reusable `.zip` knowledge packs (scripts +
//!   references) and run/build projects with preview.

pub mod config;
pub mod llm;
pub mod memory;
pub mod modes;
pub mod orchestrate;
pub mod registry;
pub mod router;
pub mod session;
pub mod skills;
pub mod tools;
pub mod types;
pub mod usage;

pub use config::AgentConfig;
pub use llm::LlmConfig;
pub use memory::MemoryStore;
pub use registry::Registry;
pub use session::Session;
pub use usage::{RateLimitConfig, TokenUsage, UsageMeter, UsageStats};
pub use types::{
    Agent, AgentOutcome, AgentTurn, Capability, Message, Mode, MsgRole, Role, Skill, TurnEvent,
    Workflow,
};
pub use tools::ToolCtx;

use crate::agent::types::Role as RoleType;
use crate::GenError;

/// Outcome of a single `turn`.
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// Whether we answered directly or orchestrated a team.
    pub orchestrated: bool,
    /// The orchestration mode used (or "direct").
    pub mode: String,
    /// Final answer to the user.
    pub answer: String,
    /// Agent-by-agent transcript for observability.
    pub transcript: std::sync::Arc<std::vec::Vec<AgentTurn>>,
    /// The plan used (mode + roster).
    pub plan: String,
}

/// Shared runtime context handed to orchestration modes.
pub struct Runner {
    pub config: AgentConfig,
    pub llm: LlmConfig,
    pub memory: MemoryStore,
    pub registry: Registry,
    /// Current session scope (user id) for memory/tool context.
    pub scope: String,
}

impl Runner {
    pub fn tool_ctx(&self) -> ToolCtx {
        ToolCtx {
            config: self.config.clone(),
            memory: self.memory.clone(),
            registry: self.registry.clone(),
            scope: self.scope.clone(),
        }
    }
}

/// The multi-agent engine. Holds shared configuration, memory and the asset
/// registry; orchestrates sessions.
#[derive(Clone)]
pub struct AgentEngine {
    pub config: AgentConfig,
    memory: MemoryStore,
    registry: Registry,
}

impl AgentEngine {
    /// Build an engine from an `AgentConfig`.
    pub fn new(config: AgentConfig) -> Result<Self, GenError> {
        std::fs::create_dir_all(&config.sessions_dir).ok();
        std::fs::create_dir_all(&config.skills_dir).ok();
        std::fs::create_dir_all(&config.root_dir).ok();
        let memory = MemoryStore::open(&config.memory_file)?;
        let registry = Registry::open(Some(&config.skills_dir.join("registry.json")))?;
        Ok(Self { config, memory, registry })
    }

    /// Start an engine from environment variables (see `AgentConfig::from_env`).
    pub fn from_env() -> Result<Self, GenError> {
        Self::new(AgentConfig::from_env())
    }

    pub fn memory(&self) -> MemoryStore {
        self.memory.clone()
    }

    pub fn registry(&self) -> Registry {
        self.registry.clone()
    }

    /// Create a new, independent session for a user scope.
    pub fn new_session(&self, scope: &str) -> Result<Session, GenError> {
        Session::new(scope, &self.config.sessions_dir)
    }

    /// Open an existing session.
    pub fn open_session(&self, id: &str) -> Result<Session, GenError> {
        Session::open(id, &self.config.sessions_dir)
    }

    /// Build runnable agents from the registry (default roster).
    pub fn build_agents(&self) -> Vec<Agent> {
        let roles = self.registry.roles();
        let caps = self.registry.capabilities();
        let skills = self.registry.skills();
        roles
            .into_iter()
            .map(|r| {
                let capabilities: Vec<Capability> = r
                    .capabilities
                    .iter()
                    .filter_map(|c| caps.iter().find(|x| &x.id == c).cloned())
                    .collect();
                let agent_skills: Vec<Skill> = r
                    .skills
                    .iter()
                    .filter_map(|s| skills.iter().find(|x| &x.id == s).cloned())
                    .collect();
                let base = r.id.clone();
                Agent {
                    id: base.clone(),
                    name: r.name.clone(),
                    seed: r.persona.clone(),
                    role: r,
                    capabilities,
                    skills: agent_skills,
                    model: self.config.llm_model.clone(),
                }
            })
            .collect()
    }

    /// Resolve a single agent by id from the default roster.
    pub fn agent(&self, id: &str) -> Option<Agent> {
        self.build_agents().into_iter().find(|a| a.id == id)
    }

    fn runner(&self, scope: &str) -> Runner {
        Runner {
            config: self.config.clone(),
            llm: self.config.llm(),
            memory: self.memory.clone(),
            registry: self.registry.clone(),
            scope: scope.to_string(),
        }
    }

    /// Run a multi-agent workflow explicitly (advanced API).
    pub async fn run_workflow(
        &self,
        scope: &str,
        workflow: &Workflow,
        agents: &[Agent],
    ) -> Result<(String, Vec<AgentTurn>), GenError> {
        let runner = self.runner(scope);
        orchestrate::run_workflow(&runner, workflow, agents).await
    }

    /// **Direct conversation**: process one user message within a session.
    ///
    /// 1. Append the user message.
    /// 2. The director decides direct vs. orchestrate (heuristic + optional LLM).
    /// 3. Run the plan (direct assistant call or a team workflow).
    /// 4. Append the assistant answer and persist.
    pub async fn turn(&self, session: &mut Session, user_input: &str) -> Result<TurnResult, GenError> {
        let mut all_msgs = session.history().to_vec();
        all_msgs.push(Message {
            role: MsgRole::User,
            name: "user".into(),
            content: user_input.to_string(),
            meta: None,
        });
        session.push(Message {
            role: MsgRole::User,
            name: "user".into(),
            content: user_input.to_string(),
            meta: None,
        })?;

        let agents = self.build_agents();
        let runner = self.runner(&session.scope);

        // Decide the plan.
        let workflow = {
            let heuristic = router::decide_heuristic(user_input, all_msgs.len(), &agents);
            match heuristic {
                router::Intent::Direct => None,
                router::Intent::Orchestrate(wf) => {
                    let plan_req = types::PlanRequest {
                        session_id: session.id.clone(),
                        user_input: user_input.to_string(),
                        history: all_msgs.clone(),
                        available_agents: agents.clone(),
                        available_skills: self.registry.skills(),
                    };
                    router::maybe_llm_plan(&runner.llm, &runner.scope, &plan_req).await.or(Some(wf))
                }
            }
        };

        match workflow {
            None => {
                // Direct: primary assistant answers.
                let primary = agents
                    .iter()
                    .find(|a| a.id == registry::roles::ASSISTANT)
                    .or_else(|| agents.first())
                    .cloned()
                    .unwrap_or_else(|| Agent {
                        id: "assistant".into(),
                        name: "主助手 (Assistant)".into(),
                        seed: "你是主助手，直接回答用户问题。".into(),
                        role: RoleType {
                            id: "assistant".into(),
                            name: "主助手".into(),
                            persona: "你是主助手，直接回答用户问题。".into(),
                            capabilities: vec!["conversation".into()],
                            skills: vec![],
                            prefer_mode: None,
                        },
                        capabilities: vec![Capability {
                            id: "conversation".into(),
                            name: "conversation".into(),
                            description: "direct dialogue".into(),
                        }],
                        skills: vec![],
                        model: self.config.llm_model.clone(),
                    });
                let answer = if self.config.mock {
                    format!("[mock·{0}] {1}", primary.name, user_input)
                } else {
                    let context = self.memory.dump(&session.scope);
                    let mut sys = modes::agent_system_prompt(&primary);
                    if !context.is_empty() {
                        sys.push_str(&format!("\n关于用户的记忆：\n{context}"));
                    }
                    let mut msgs = session.recent(16).to_vec();
                    if msgs.is_empty() {
                        msgs.push(Message {
                            role: MsgRole::User,
                            name: "user".into(),
                            content: user_input.to_string(),
                            meta: None,
                        });
                    }
                    runner.llm.complete(&runner.scope, &sys, &msgs).await?
                };
                let a = answer.clone();
                session.push(Message {
                    role: MsgRole::Agent,
                    name: primary.name.clone(),
                    content: a,
                    meta: None,
                })?;
                Ok(TurnResult {
                    orchestrated: false,
                    mode: "direct".into(),
                    plan: "direct answer by primary assistant".into(),
                    transcript: std::sync::Arc::new(vec![AgentTurn {
                        agent: primary.name.clone(),
                        content: answer.clone(),
                        tools: vec![],
                    }]),
                    answer,
                })
            }
            Some(wf) => {
                let (out, transcript) = orchestrate::run_workflow(&runner, &wf, &agents).await?;
                let a = out.clone();
                session.push(Message {
                    role: MsgRole::Agent,
                    name: "team".into(),
                    content: a,
                    meta: None,
                })?;
                // Optionally persist a memory hint (the gist as "last_topic").
                let gist = out.chars().take(200).collect::<String>();
                let _ = self.memory.remember(&session.scope, "last_topic", serde_json::json!(gist));
                Ok(TurnResult {
                    orchestrated: true,
                    mode: wf.mode.label().to_string(),
                    plan: wf.rationale.clone(),
                    transcript: std::sync::Arc::new(transcript),
                    answer: out,
                })
            }
        }
    }

    /// **Asset creation**: parse natural-language request and create roles /
    /// capabilities / skills / persona. Uses the LLM when available; otherwise
    /// falls back to heuristics for role creation.
    pub async fn craft_assets(&self, scope: &str, text: &str) -> Result<String, GenError> {
        let runner = self.runner(scope);
        if !runner.llm.api_key.is_empty() && !self.config.mock {
            let system = "你是资产生成器。把用户对智能体角色/技能/能力/人设的描述解析为JSON数组，每项形如：\
            {\"type\":\"role|capability|skill\",\"id\":\"<英文id>\",\"name\":\"<中文名>\",\"description\":\"\",\"persona\":\"<仅role>\",\"capabilities\":[\"conversation\"],\"skills\":[]}。\
            只输出JSON数组。";
            let resp = runner.llm.chat(&runner.scope, system, text).await?;
            if let Ok(arr) = serde_json::from_str::<Vec<AssetSpec>>(resp.trim_matches('`').trim()) {
                return self.apply_asset_specs(scope, &arr);
            }
            return Err(GenError::Backend("assets", "无法解析资产生成结果".into()));
        }
        // Heuristic fallback: `角色：X（人设）`
        Ok(self.heuristic_craft(text))
    }

    fn apply_asset_specs(&self, _scope: &str, specs: &[AssetSpec]) -> Result<String, GenError> {
        let mut created: Vec<String> = Vec::new();
        for s in specs {
            match s.etype.as_str() {
                "capability" => {
                    self.registry.create_capability(Capability {
                        id: s.id.clone(),
                        name: s.name.clone(),
                        description: s.description.clone(),
                    })?;
                    created.push(format!("能力「{}」", s.name));
                }
                "skill" => {
                    self.registry.create_skill(Skill {
                        id: s.id.clone(),
                        name: s.name.clone(),
                        version: "0.1.0".into(),
                        description: s.description.clone(),
                        instruction: s.persona.clone(),
                        scripts: vec![],
                        references: vec![],
                        needs_tools: s.capabilities.clone(),
                    })?;
                    created.push(format!("技能「{}」", s.name));
                }
                _ => {
                    self.registry.create_role(RoleType {
                        id: s.id.clone(),
                        name: s.name.clone(),
                        persona: s.persona.clone(),
                        capabilities: s.capabilities.clone(),
                        skills: s.skills.clone(),
                        prefer_mode: None,
                    })?;
                    created.push(format!("角色「{}」", s.name));
                }
            }
        }
        if created.is_empty() {
            Ok("（未创建任何资产）".to_string())
        } else {
            Ok(format!("已创建：{}", created.join("、")))
        }
    }

    fn heuristic_craft(&self, text: &str) -> String {
        let lower = text.to_lowercase();
        if lower.contains("角色") && text.contains('：') {
            let name = text
                .split('：')
                .nth(1)
                .unwrap_or("未命名")
                .trim()
                .trim_matches('。')
                .to_string();
            let id = if name.is_empty() {
                "角色".to_string()
            } else {
                name.chars().take(8).collect()
            };
            let _ = self.registry.create_role(RoleType {
                id,
                name: name.clone(),
                persona: text.to_string(),
                capabilities: vec!["conversation".into(), "research".into()],
                skills: vec![],
                prefer_mode: None,
            });
            format!("已创建角色「{name}」（离线启发式解析；如需更精细可配置 LLM 后重试）")
        } else {
            "请输入要创建的角色/技能/能力描述，例如「创建一个善于UX设计的角色，擅长分析用户旅程」。".to_string()
        }
    }
}

/// Deserialization shape for asset-craft LLM output.
#[derive(Debug, serde::Deserialize)]
pub struct AssetSpec {
    #[serde(rename = "type", default)]
    pub etype: String,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub persona: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub skills: Vec<String>,
}

impl Clone for AssetSpec {
    fn clone(&self) -> Self {
        Self {
            etype: self.etype.clone(),
            id: self.id.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            persona: self.persona.clone(),
            capabilities: self.capabilities.clone(),
            skills: self.skills.clone(),
        }
    }
}

// Re-export for convenience.
pub use skills::import_skill_zip;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn mock_engine() -> (AgentEngine, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("aiapp_agent_test_{}", std::process::id()));
        let mut cfg = AgentConfig::default_with_root(dir.clone());
        cfg.mock = true;
        let engine = AgentEngine::new(cfg).expect("engine");
        (engine, dir)
    }

    #[test]
    fn direct_conversation_in_mock() {
        let (engine, _) = mock_engine();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut session = engine.new_session("user1").unwrap();
        let t = rt.block_on(engine.turn(&mut session, "你好")).unwrap();
        assert!(!t.orchestrated);
        assert!(t.answer.contains("你好"));
        assert_eq!(session.history().len(), 2);
    }

    #[test]
    fn orchestration_runs_in_mock() {
        let (engine, _) = mock_engine();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut session = engine.new_session("user1").unwrap();
        let t = rt
            .block_on(engine.turn(&mut session, "帮我生成一个 Todo 应用"))
            .unwrap();
        assert!(t.orchestrated, "should orchestrate an execution task");
        // mock answers are non-empty transcripts
        assert!(!t.transcript.is_empty());
    }

    #[test]
    fn all_six_modes_run_in_mock() {
        let (engine, _) = mock_engine();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let agents = engine.build_agents();
        for mode in [
            Mode::Sequential,
            Mode::Parallel,
            Mode::SupervisorWorker,
            Mode::Debate,
            Mode::GroupChat,
            Mode::ToolLoop,
        ] {
            let wf = Workflow {
                mode,
                agent_ids: agents.iter().map(|a| a.id.clone()).collect(),
                tasks: vec!["做一个番茄钟".into()],
                rationale: "smoke".into(),
            };
            let (out, tx) = rt
                .block_on(engine.run_workflow("user1", &wf, &agents))
                .unwrap();
            assert!(!out.is_empty());
            assert!(!tx.is_empty());
        }
    }

    #[test]
    fn asset_creation_heuristic() {
        let (engine, _) = mock_engine();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let out = rt.block_on(engine.craft_assets("user1", "创建角色：测试工程师")).unwrap();
        assert!(out.contains("测试工程师"));
        assert!(engine.agent("测试工程师").is_some());
    }

    #[test]
    fn skill_import_zip() {
        let (engine, dir) = mock_engine();
        let cfg = engine.config.clone();
        // Build a tiny skill zip.
        let zip_path = dir.join("myskill.zip");
        let mut w = zip::ZipWriter::new(std::fs::File::create(&zip_path).unwrap());
        w.start_file("SKILL.md", zip::write::SimpleFileOptions::default()).unwrap();
        w.write_all(
            "---\nname: 代码审查\ndescription: 审查产出\ninstruction: 总是给出问题清单\n---\n"
                .as_bytes(),
        )
        .unwrap();
        w.finish().unwrap();
        let skill = super::skills::import_skill_zip(&zip_path, &cfg, &engine.registry()).unwrap();
        assert_eq!(skill.name, "代码审查");
        assert!(engine.registry().skill(&skill.id).is_some());
    }
}
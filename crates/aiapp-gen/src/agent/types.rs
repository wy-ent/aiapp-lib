//! Core asset model for the multi-agent engine: messages, capabilities,
//! skills, roles (personas) and runnable agents.
//!
//! Asset creation is one of the engine's capabilities: roles, capabilities,
//! skills and persona can be produced through natural conversation and stored
//! in the registry (see `registry.rs`).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Who produced a message inside a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MsgRole {
    /// Human (the end user).
    User,
    /// An agent / assistant producing content.
    Agent,
    /// Tool execution result fed back into the loop.
    Tool,
    /// System-level prompt / context.
    System,
}

/// A single message inside a session's conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MsgRole,
    /// Producer identity: `user`, an agent name, or a tool name.
    pub name: String,
    pub content: String,
    /// Optional structured metadata (e.g. tool call id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

/// A reusable capability that an agent may claim (e.g. `code`, `web_search`, `file_ops`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub id: String,
    pub name: String,
    pub description: String,
}

/// A skill: reusable knowledge/instruction package that can be imported from a
/// `.zip` (see `skills.rs`) and attached to agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    /// System-level instruction injected into the agent's system prompt.
    #[serde(default)]
    pub instruction: String,
    /// Relative script entry points shipped inside the skill package.
    #[serde(default)]
    pub scripts: Vec<String>,
    /// Reference material paths shipped inside the skill package.
    #[serde(default)]
    pub references: Vec<String>,
    /// Tools this skill expects to be available.
    #[serde(default)]
    pub needs_tools: Vec<String>,
}

/// A role / persona: packaged as a reusable asset. An agent is an instantiation
/// of a role inside a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub name: String,
    /// Personality / system persona text.
    #[serde(default)]
    pub persona: String,
    /// Capabilities this role can use.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Skills attached to this role.
    #[serde(default)]
    pub skills: Vec<String>,
    /// The orchestration mode this role prefers, if any.
    #[serde(default)]
    pub prefer_mode: Option<String>,
}

/// A runnable agent obtained by instantiating a `Role` within a session.
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub role: Role,
    /// Effective capabilities (union of role capabilities + session-kind).
    pub capabilities: Vec<Capability>,
    pub skills: Vec<Skill>,
    pub model: String,
    /// Any session-local role-specific seed prompt.
    pub seed: String,
}

impl Agent {
    /// Build an agent message.
    pub fn msg(&self, content: impl Into<String>) -> Message {
        Message {
            role: MsgRole::Agent,
            name: self.name.clone(),
            content: content.into(),
            meta: None,
        }
    }
}

/// The 6 standard orchestration modes used to assemble a collaborative workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    /// 1. Sequential pipeline: agents run one after another, each output feeds the next.
    Sequential,
    /// 2. Parallel fan-out: a coordinator splits the task, workers run concurrently, results merge.
    Parallel,
    /// 3. Hierarchical supervisor-worker: a supervisor delegates to workers then critiques/merges.
    SupervisorWorker,
    /// 4. Debate / critic: a producer and a reviewer iterate over rounds.
    Debate,
    /// 5. Group chat: a set of agents exchange messages in rounds until consensus or deadline.
    GroupChat,
    /// 6. Tool loop / auto: a single primary agent iterates with tool calls until done.
    ToolLoop,
}

impl Mode {
    /// Parse from a string (case-insensitive), with Chinese aliases.
    pub fn parse(s: &str) -> Option<Mode> {
        let t = s.trim().to_lowercase();
        Some(match t.as_str() {
            "sequential" | "pipeline" | "chain" | "顺序" | "流水线" => Mode::Sequential,
            "parallel" | "fanout" | "并行" | "扇出" => Mode::Parallel,
            "supervisor" | "hierarchical" | "worker" | "主管" | "分层" => Mode::SupervisorWorker,
            "debate" | "critic" | "review" | "辩论" | "评审" => Mode::Debate,
            "group" | "groupchat" | "chat" | "群聊" => Mode::GroupChat,
            "tool_loop" | "tool-loop" | "autoloop" | "auto" | "工具链" | "自动" => Mode::ToolLoop,
            _ => return None,
        })
    }

    /// Display name.
    pub fn label(self) -> &'static str {
        match self {
            Mode::Sequential => "sequential",
            Mode::Parallel => "parallel",
            Mode::SupervisorWorker => "supervisor_worker",
            Mode::Debate => "debate",
            Mode::GroupChat => "group_chat",
            Mode::ToolLoop => "tool_loop",
        }
    }
}

/// A plan produced by the director: which mode, which agents, and subtask specs.
#[derive(Debug, Clone)]
pub struct Workflow {
    pub mode: Mode,
    /// Ordered agent ids (sequential / group / pipeline order).
    pub agent_ids: Vec<String>,
    /// Subtasks; filled for parallel / supervisor workflows.
    pub tasks: Vec<String>,
    /// Free-form rationale for observability.
    pub rationale: String,
}

/// An agent execution result.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    pub agent_name: String,
    pub content: String,
    pub tool_calls: Vec<String>,
}

/// Configuration for the director/planner LLM call.
#[derive(Debug, Clone)]
pub struct PlanRequest {
    pub session_id: String,
    pub user_input: String,
    pub history: Vec<Message>,
    pub available_agents: Vec<Agent>,
    pub available_skills: Vec<Skill>,
}

/// A turn-level event used to report transcripts to the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEvent {
    /// Orchestration mode label for this turn.
    pub mode: String,
    /// Ordered list of (agent_name, message) produced during the turn.
    pub transcript: Vec<AgentTurn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTurn {
    pub agent: String,
    pub content: String,
    /// Tool names invoked by this agent this turn (observability).
    #[serde(default)]
    pub tools: Vec<String>,
}

/// Blob typed by current user's memory store.
pub type MemoryMap = HashMap<String, serde_json::Value>;
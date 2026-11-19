//! Director / router: decides whether to answer directly or orchestrate a
//! multi-agent workflow, which mode to use, and which agents to assemble.
//!
//! Two layers:
//! 1. A fast, offline heuristic that always returns a sensible plan (works in
//!    mock mode and when no LLM key is configured).
//! 2. An optional LLM refinement step that replaces/confirms the plan.

use crate::agent::types::{Agent, Mode, Workflow};
use serde::Deserialize;

/// Consolidated intent for a piece of user input.
#[derive(Debug, Clone)]
pub enum Intent {
    /// Answer directly with the primary assistant (no team).
    Direct,
    /// Run a collaborative multi-agent workflow.
    Orchestrate(Workflow),
}

/// A plan produced by the LLM / heuristic, parsed into a `Workflow`.
#[derive(Debug, Deserialize)]
struct PlanSpec {
    #[serde(default)]
    mode: String,
    #[serde(default)]
    agents: Vec<String>,
    #[serde(default)]
    tasks: Vec<String>,
    #[serde(default)]
    rationale: String,
    #[serde(default)]
    direct: bool,
}

/// Heuristic signal words that indicate a "make/do/build" task → orchestrate.
const EXECUTION_HINTS: [&str; 32] = [
    "做", "构建", "建", "造", "创建", "生成", "开发", "写一个", "写个", "搭建", "实现",
    "设计一个", "生成一个", "应用", "app", "网站", "页面", "小程序", "开始", "启动",
    "build", "create", "make", "develop", "implement", "write", "generate",
    "scaffold", "craft", "produce", "设计", "demo",
];

/// Choose a default orchestration mode by heuristics over the input.
pub fn default_mode_for(input: &str) -> Mode {
    let lower = input.to_lowercase();
    // Debate/review for improvement or quality-critical asks.
    if ["优化", "改进", "打磨", "评审", "review", "improve", "polish", "refine"]
        .iter()
        .any(|kw| input.contains(kw))
    {
        return Mode::Debate;
    }
    // Large multi-part / list-style goals → supervisor-worker.
    if ["同时", "以及", "都要", "整体", "系统", "项目", "平台", "place都"]
        .iter()
        .any(|kw| input.contains(kw))
    {
        return Mode::SupervisorWorker;
    }
    // Autonomous repeated work → tool loop.
    if ["自动", "持续", "循环", "反复", "auto", "loop", "iterate", "直至", "直到"]
        .iter()
        .any(|kw| lower.contains(kw))
    {
        return Mode::ToolLoop;
    }
    // Explicitly fast / many independent parts → parallel.
    if ["快速", "尽快", "多线程", "并行", "快", "parallel", "fast", "同时处理", "批量"]
        .iter()
        .any(|kw| lower.contains(kw))
    {
        return Mode::Parallel;
    }
    // Group/collective brainstorming.
    if ["头脑风暴", "讨论", "想法", "多个观点", "brainstorm", "opinion", "观点"]
        .iter()
        .any(|kw| lower.contains(kw))
    {
        return Mode::GroupChat;
    }
    // Default: pipeline (most common single goal → sequential steps).
    Mode::Sequential
}

fn is_execution(input: &str) -> bool {
    let lower = input.trim().to_lowercase();
    EXECUTION_HINTS.iter().any(|h| lower.contains(h))
}

/// Offline heuristic planner. Returns `Direct` for pure dialogue, else an
/// orchestration workflow with a hand-picked roster.
pub fn decide_heuristic(
    user_input: &str,
    user_history_len: usize,
    agents: &[Agent],
) -> Intent {
    if !is_execution(user_input) && user_history_len == 0 && agents_any_convo(agents) {
        return Intent::Direct;
    }
    if !is_execution(user_input) {
        // A question without execution intent still goes to the assistant,
        // but a group chat broadens perspective for opinion-type asks.
        let mode = default_mode_for(user_input);
        if matches!(mode, Mode::GroupChat | Mode::Debate) {
            return Intent::Orchestrate(Workflow {
                mode,
                agent_ids: roster_infer_ids(agents, None),
                tasks: vec![user_input.to_string()],
                rationale: "opinion/brainstorm task routed to a team".into(),
            });
        }
        return Intent::Direct;
    }

    let mode = default_mode_for(user_input);
    let user_mentioned = extract_agent_refs(user_input, agents);
    let ids = roster_infer_ids(agents, user_mentioned);
    Intent::Orchestrate(Workflow {
        mode,
        agent_ids: ids,
        tasks: vec![user_input.to_string()],
        rationale: format!("routed to {:?} workflow", mode),
    })
}

fn agents_any_convo(agents: &[Agent]) -> bool {
    agents.iter().any(|a| a.capabilities.iter().any(|c| c.id == "conversation"))
}

fn extract_agent_refs(input: &str, agents: &[Agent]) -> Option<String> {
    let lower = input.to_lowercase();
    agents.iter().find_map(|a| {
        let name = a.name.to_lowercase();
        if name != "导演" && (lower.contains(&a.id) || lower.contains(&name)) {
            Some(a.id.clone())
        } else {
            None
        }
    })
}

/// Pick a roster of agent ids based on the mode.
fn roster_infer_ids(agents: &[Agent], user_mentioned: Option<String>) -> Vec<String> {
    if let Some(m) = user_mentioned {
        if agents.iter().any(|a| a.id == m) {
            return vec![m];
        }
    }
    // Default roster by role id availability.
    let ids: Vec<String> = agents.iter().map(|a| a.id.clone()).collect();
    if ids.is_empty() {
        return ids;
    }
    ids
}

/// Optional LLM-based plan refinement. When parsing fails or themodel is
/// unavailable, returns `None` (caller falls back to the heuristic).
pub async fn maybe_llm_plan(
    llm: &crate::agent::llm::LlmConfig,
    scope: &str,
    request: &crate::agent::types::PlanRequest,
) -> Option<Workflow> {
    if llm.api_key.is_empty() {
        return None;
    }
    let agents_desc: Vec<String> = request
        .available_agents
        .iter()
        .map(|a| format!("- {} (id: {})", a.name, a.id))
        .collect();
    let skills_desc: Vec<String> = request
        .available_skills
        .iter()
        .map(|s| format!("- {} (id: {})", s.name, s.id))
        .collect();
    let system = "你是编排导演。判断用户意图：如果只是提问/分析就输出 {\"direct\":true}；如果要做成一件事(开发/构建/生成应用等)，输出一个协作工作流JSON：\
    {\"mode\":\"sequential|parallel|supervisor_worker|debate|group_chat|tool_loop\",\"agents\":[\"<agent_id>\"...],\"tasks\":[\"<子任务>\"...],\"rationale\":\"<理由>\",\"direct\":false}。\
    只能使用给出的 agent id；团队要精选(2-5个)。只输出JSON。";
    let user = format!(
        "用户输入：{}\n\n可用Agent：\n{}\n\n可用技能：\n{}",
        request.user_input,
        agents_desc.join("\n"),
        skills_desc.join("\n")
    );
    let resp = llm.chat(scope, system, &user).await.ok()?;
    let cleaned = resp
        .trim()
        .trim_start_matches("```")
        .trim_start_matches("json")
        .trim()
        .trim_end_matches("```")
        .trim();
    let spec: PlanSpec = serde_json::from_str(cleaned).ok()?;
    if spec.direct {
        return None;
    }
    let mode = Mode::parse(&spec.mode)?;
    Some(Workflow {
        mode,
        agent_ids: spec.agents,
        tasks: spec.tasks,
        rationale: spec.rationale,
    })
}
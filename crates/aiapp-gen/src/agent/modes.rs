//! The 6 orchestration modes. Each produces a final answer plus a transcript
//! of (agent, content) turns for observability.

use crate::agent::types::{Agent, AgentTurn, Message, MsgRole};
use crate::GenError;
use futures::future::join_all;

use super::Runner;

/// Build the system prompt for an agent from its role + capabilities + skills.
pub fn agent_system_prompt(agent: &Agent) -> String {
    let caps: Vec<String> = agent.capabilities.iter().map(|c| c.name.clone()).collect();
    let mut p = format!(
        "你是智能体「{}」。\n{}\n\n你可用的能力/工具：{}\n",
        agent.name,
        agent.role.persona,
        caps.join(", ")
    );
    if !agent.seed.is_empty() {
        p.push_str(&format!("\n额外指示：{}\n", agent.seed));
    }
    if !agent.skills.is_empty() {
        p.push_str("\n你携带的技能与指导：\n");
        for sk in &agent.skills {
            if !sk.instruction.is_empty() {
                p.push_str(&format!("- [{}] {}\n", sk.name, sk.instruction));
            }
        }
    }
    p
}

fn terminal_prompt() -> &'static str {
    "请直接产出最终结果（不要解释如何完成）。"
}

/// Resolve the roster agents for a workflow; falls back to all agents.
fn resolve_roster<'a>(agents: &'a [Agent], ids: &[String]) -> Vec<&'a Agent> {
    if ids.is_empty() {
        return agents.iter().collect();
    }
    let mut out = Vec::new();
    for id in ids {
        if let Some(a) = agents.iter().find(|a| &a.id == id) {
            out.push(a);
        }
    }
    if out.is_empty() {
        agents.iter().collect()
    } else {
        out
    }
}

/// Small helper to call an agent once against a set of input messages.
async fn run_agent(
    runner: &Runner,
    agent: &Agent,
    msgs: Vec<Message>,
) -> Result<String, GenError> {
    if runner.config.mock {
        return Ok(mock_reply(agent, &msgs));
    }
    runner.llm.complete(&runner.scope, &agent_system_prompt(agent), &msgs).await
}

fn mock_reply(a: &Agent, msgs: &[Message]) -> String {
    let last = msgs.iter().rev().find(|m| m.role != MsgRole::Tool);
    let subject = last.map(|m| m.content.chars().take(40).collect::<String>()).unwrap_or_default();
    format!("[mock·{}] 已接收任务：{}", a.name, subject)
}

fn tell(name: &str, content: impl Into<String>, tools: Vec<String>) -> AgentTurn {
    AgentTurn { agent: name.into(), content: content.into(), tools }
}

/// 1. Sequential pipeline: each agent's output becomes the next agent's input.
pub async fn run_sequential(
    runner: &Runner,
    agents: &[Agent],
    ids: &[String],
    task: &str,
) -> Result<(String, Vec<AgentTurn>), GenError> {
    let roster = resolve_roster(agents, ids);
    let mut transcript = Vec::new();
    let mut carry = task.to_string();
    for agent in roster {
        let msgs = vec![Message {
            role: MsgRole::User,
            name: "director".into(),
            content: carry,
            meta: None,
        }];
        let out = run_agent(runner, agent, msgs).await?;
        transcript.push(tell(&agent.name, &out, Vec::new()));
        carry = out;
    }
    Ok((carry, transcript))
}

/// 2. Parallel fan-out: one coordinator splits the task into N subtasks, workers
/// run concurrently, results are merged by the coordinator.
pub async fn run_parallel(
    runner: &Runner,
    agents: &[Agent],
    ids: &[String],
    task: &str,
    subtasks: &[String],
) -> Result<(String, Vec<AgentTurn>), GenError> {
    let roster = resolve_roster(agents, ids);
    let mut transcript = Vec::new();

    // Coordinator splits into subtasks if none provided.
    let subs: Vec<String> = if subtasks.is_empty() {
        split_into_subs(runner, roster.len(), task).await?
    } else {
        subtasks.to_vec()
    };

    // Run workers concurrently.
    let jobs: Vec<_> = roster
        .iter()
        .enumerate()
        .map(|(i, agent)| {
            let content = subs.get(i).cloned().unwrap_or_else(|| task.to_string());
            let t = format!("[子任务 {}] {}", i + 1, content);
            let msgs = vec![Message {
                role: MsgRole::User,
                name: "coordinator".into(),
                content: t,
                meta: None,
            }];
            run_agent(runner, agent, msgs)
        })
        .collect();
    let results = join_all(jobs).await;
    let mut merged = String::new();
    for (i, r) in results.into_iter().enumerate() {
        let agent = roster[i];
        let out = r.map_err(|e| GenError::Backend("mode.parallel", e.to_string()))?;
        transcript.push(tell(&agent.name, &out, Vec::new()));
        merged.push_str(&format!("\n\n===== 子任务 {} 结果 =====\n{}", i + 1, out));
    }

    // Merge via the first agent (or a designated coordinator).
    if let Some(coordinator) = roster.first() {
        let msgs = vec![Message {
            role: MsgRole::User,
            name: "coordinator".into(),
            content: format!("汇总以下并行结果为一个统一的最终交付。\n{}", merged),
            meta: None,
        }];
        let final_out = run_agent(runner, coordinator, msgs).await?;
        transcript.push(tell(&coordinator.name, &final_out, Vec::new()));
        return Ok((final_out, transcript));
    }
    Ok((merged, transcript))
}

async fn split_into_subs(
    runner: &Runner,
    n: usize,
    task: &str,
) -> Result<Vec<String>, GenError> {
    if n <= 1 {
        return Ok(vec![task.to_string()]);
    }
    if runner.config.mock || runner.llm.api_key.is_empty() {
        return Ok(vec![task.to_string(); n]);
    }
    let system = "你是任务拆解者。把任务拆成多个相互独立、可并行的子任务，每行一个，用编号开头。只输出子任务，不解释。";
    let out = runner.llm.chat(&runner.scope, system, task).await?;
    let pieces: Vec<String> = out
        .lines()
        .map(|l| l.trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ').to_string())
        .filter(|l| !l.is_empty())
        .collect();
    if pieces.is_empty() {
        Ok(vec![task.to_string(); n])
    } else {
        Ok(pieces)
    }
}

/// 3. Hierarchical supervisor-worker: supervisor delegates, then critiques + merges.
pub async fn run_supervisor_worker(
    runner: &Runner,
    agents: &[Agent],
    ids: &[String],
    task: &str,
) -> Result<(String, Vec<AgentTurn>), GenError> {
    let roster = resolve_roster(agents, ids);
    let mut transcript = Vec::new();
    let (supervisor_idx, workers) = if roster.len() > 1 {
        (0, &roster[1..])
    } else {
        (0, roster.as_slice())
    };
    let supervisor = roster[supervisor_idx];

    let subs = split_into_subs(runner, workers.len().max(1), task).await?;
    let jobs: Vec<_> = workers
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let t = format!("[任务分工] {}", subs.get(i).cloned().unwrap_or_else(|| task.to_string()));
            let msgs = vec![Message {
                role: MsgRole::User,
                name: "supervisor".into(),
                content: t,
                meta: None,
            }];
            run_agent(runner, *w, msgs)
        })
        .collect();
    let results = join_all(jobs).await;

    let mut merged = String::new();
    for (i, r) in results.into_iter().enumerate() {
        let w = workers[i];
        let out = r.map_err(|e| GenError::Backend("mode.supervisor", e.to_string()))?;
        transcript.push(tell(&w.name, &out, Vec::new()));
        merged.push_str(&format!("\n===== 工人 {} 产出 =====\n{}", i + 1, out));
    }

    let final_msgs = vec![
        Message {
            role: MsgRole::System,
            name: "system".into(),
            content: "你是主管。请评审下属产出，修正错误、补齐缺口，汇总为最终交付。".into(),
            meta: None,
        },
        Message {
            role: MsgRole::User,
            name: "supervisor".into(),
            content: format!("原始任务：{}\n\n下属产出：{}", task, merged),
            meta: None,
        },
    ];
    let final_out = run_agent(runner, supervisor, final_msgs).await?;
    transcript.push(tell(&supervisor.name, &final_out, Vec::new()));
    Ok((final_out, transcript))
}

/// 4. Debate / critic: producer writes, reviewer critiques, iterate rounds.
pub async fn run_debate(
    runner: &Runner,
    agents: &[Agent],
    ids: &[String],
    task: &str,
    rounds: usize,
) -> Result<(String, Vec<AgentTurn>), GenError> {
    let roster = resolve_roster(agents, ids);
    let rounds = rounds.max(1);
    let (producer, reviewer) = if roster.len() > 1 {
        (roster[0], roster[1])
    } else {
        (roster[0], roster[0])
    };
    let mut transcript = Vec::new();
    let mut draft_holder = task.to_string();

    for r in 0..rounds {
        let produce_msgs = vec![
            Message {
                role: MsgRole::System,
                name: "system".into(),
                content: "你可以参考前一轮的评审反馈，改进你的产出。".into(),
                meta: None,
            },
            Message {
                role: MsgRole::User,
                name: "moderator".into(),
                content: if r == 0 { draft_holder.clone() } else { format!("结合以下评审改进你的产出。\n评审：\n{}\n\n原始任务：{}", draft_holder, task) },
                meta: None,
            },
        ];
        let draft = run_agent(runner, producer, produce_msgs).await?;
        transcript.push(tell(&producer.name, &draft, Vec::new()));

        if r + 1 < rounds {
            let review_msgs = vec![
                Message {
                    role: MsgRole::System,
                    name: "system".into(),
                    content: "你是严格的评审者，输出结构化改进意见（问题清单+具体建议）。".into(),
                    meta: None,
                },
                Message {
                    role: MsgRole::User,
                    name: "moderator".into(),
                    content: draft.clone(),
                    meta: None,
                },
            ];
            let review = run_agent(runner, reviewer, review_msgs).await?;
            transcript.push(tell(&reviewer.name, &review, Vec::new()));
            draft_holder = review;
        } else {
            draft_holder = draft;
        }
    }
    Ok((draft_holder, transcript))
}

/// 5. Group chat: agents alternate turns until a facilitator observes
/// convergence or a round limit; then a consensus is drawn.
pub async fn run_group_chat(
    runner: &Runner,
    agents: &[Agent],
    ids: &[String],
    task: &str,
    rounds: usize,
) -> Result<(String, Vec<AgentTurn>), GenError> {
    let roster = resolve_roster(agents, ids);
    if roster.is_empty() {
        return Err(GenError::Backend("mode.group_chat", "no agents".into()));
    }
    let mut transcript = Vec::new();
    let mut history: Vec<Message> = vec![Message {
        role: MsgRole::System,
        name: "system".into(),
        content: format!("这是一个多智能体协作讨论。原始任务：{}。请给出你的专业意见，并参考前_发言。", task),
        meta: None,
    }];

    for round in 0..rounds.max(1) {
        for agent in &roster {
            let msgs = history.clone();
            let out = match run_agent(runner, agent, msgs).await {
                Ok(o) => o,
                Err(e) => format!("(agent {0} error: {1})", agent.name, e),
            };
            transcript.push(tell(&agent.name, &out, Vec::new()));
            history.push(Message {
                role: MsgRole::Agent,
                name: agent.name.clone(),
                content: out,
                meta: None,
            });
            // Keep history bounded.
            if history.len() > 24 {
                history.drain(1..history.len() - 20);
            }
        }
        let _ = round;
    }

    // Consensus from the last agent in the roster.
    let consensus = run_agent(
        runner,
        roster[0],
        history.clone(),
    )
    .await
    .unwrap_or_else(|e| format!("converged (error {e})"));
    transcript.push(tell(&roster[0].name, &consensus, Vec::new()));
    Ok((consensus, transcript))
}

use crate::agent::tools::{parse_tool_block, run_tool, tool_message, ToolCall};

/// 6. Tool loop / auto: a single primary agent iterates — call a tool, read the
/// result, continue — until it decides to stop (no tool block / FINAL:).
pub async fn run_tool_loop(
    runner: &Runner,
    agents: &[Agent],
    ids: &[String],
    task: &str,
    max_steps: usize,
) -> Result<(String, Vec<AgentTurn>), GenError> {
    let roster = resolve_roster(agents, ids);
    let Some(primary) = roster.first().copied().or_else(|| agents.first()) else {
        return Ok(("(no agent available)".to_string(), Vec::new()));
    };
    let mut ctx = runner.tool_ctx();
    let mut transcript = Vec::new();
    let mut log: Vec<Message> = vec![
        Message {
            role: MsgRole::User,
            name: "user".into(),
            content: task.to_string(),
            meta: None,
        },
    ];
    let tools_used = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    let mut extra_system = format!(
        "你可以调用工具来实际完成任务。用法：输出 ```tool\n{{{{\"name\":\"<tool名>\",\"args\":{{...}}}}\n```\n\
         可用工具：file_read,file_write,file_append,file_list,file_search,web_search,fetch_url,memory_recall,memory_remember,memory_dump,run_cmd,start_preview。\n\
         完成后再输出最终结果（不要工具块）。工具名+参数示例：file_write -> {{\"path\":\"x.txt\",\"content\":\"hi\"}}。"
    );
    // inject skills instruction
    for sk in &primary.skills {
        if !sk.instruction.is_empty() {
            extra_system.push_str(&format!("\n[技能 {}] {}", sk.name, sk.instruction));
        }
    }

    for step in 0..max_steps.max(1) {
        let msgs = {
            let mut all = log.clone();
            all.insert(
                0,
                Message {
                    role: MsgRole::System,
                    name: "system".into(),
                    content: extra_system.clone(),
                    meta: None,
                },
            );
            all
        };
        let out = run_agent(runner, primary, msgs).await?;

        // Did the agent request a tool?
        let text_for_parse = out.clone();
        let maybe_tool = parse_tool_block(&text_for_parse);
        if let Some(call) = maybe_tool {
            let ToolCall { name, args } = call;
            // record usage
            tools_used.lock().unwrap().push(name.clone());
            let result = run_tool(&ctx, &name, serde_json::Value::Object(args)).await;
            transcript.push(tell(&primary.name, &format!("[tool:{name}] {result}"), vec![name.clone()]));
            log.push(tool_message(&name, result));
            // bump scope context
            ctx = runner.tool_ctx();
            continue;
        }

        // No tool block → this is the final answer.
        transcript.push(tell(&primary.name, &out, tools_used.lock().unwrap().clone()));
        log.push(Message {
            role: MsgRole::Agent,
            name: primary.name.clone(),
            content: out.clone(),
            meta: None,
        });
        return Ok((out, transcript));
    }

    Ok((
        "(tool loop reached step limit; last state logged)".to_string(),
        transcript,
    ))
}
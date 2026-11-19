//! Workflow dispatcher: given a `Workflow`, run the corresponding orchestration
//! mode over the resolved roster and return the final answer + transcript.

use crate::agent::modes;
use crate::agent::types::{Agent, Mode, Workflow};
use crate::GenError;

/// Run a workflow to completion. `agents` is the full roster; `workflow.agent_ids`
/// picks the participating subset (empty = use all).
pub async fn run_workflow(
    runner: &super::Runner,
    workflow: &Workflow,
    agents: &[Agent],
) -> Result<(String, Vec<crate::agent::types::AgentTurn>), GenError> {
    let task = workflow
        .tasks
        .first()
        .cloned()
        .unwrap_or_else(|| "（无具体任务）".to_string());

    match workflow.mode {
        Mode::Sequential => modes::run_sequential(runner, agents, &workflow.agent_ids, &task).await,
        Mode::Parallel => {
            modes::run_parallel(runner, agents, &workflow.agent_ids, &task, &workflow.tasks).await
        }
        Mode::SupervisorWorker => {
            modes::run_supervisor_worker(runner, agents, &workflow.agent_ids, &task).await
        }
        Mode::Debate => modes::run_debate(runner, agents, &workflow.agent_ids, &task, 3).await,
        Mode::GroupChat => modes::run_group_chat(runner, agents, &workflow.agent_ids, &task, 2).await,
        Mode::ToolLoop => modes::run_tool_loop(runner, agents, &workflow.agent_ids, &task, 8).await,
    }
}
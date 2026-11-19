//! Async, OpenAI-compatible Chat Completions client built on `reqwest`.
//!
//! Used by every agent in the orchestration engine. The transport is fully
//! async so that parallel fan-out / group-chat modes can call the model
//! concurrently (see `modes.rs`).

use crate::agent::types::{Message, MsgRole};
use crate::agent::usage::{TokenUsage, UsageMeter};
use crate::GenError;
use serde::Deserialize;
use std::sync::Arc;

/// Credentials + model settings for a given LLM provider.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// OpenAI-compatible base URL, e.g. `https://api.openai.com/v1`.
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout_secs: u64,
    /// Engine-wide meter enforcing rate limits and accumulating token usage.
    pub meter: Arc<UsageMeter>,
}

#[derive(Debug, Deserialize)]
struct ChatResp {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMsg,
}

#[derive(Debug, Deserialize)]
struct ChatMsg {
    content: Option<String>,
}

/// Map our `Message` type to the OpenAI wire messages, translating tool results.
fn to_wire(messages: &[Message]) -> Vec<serde_json::Value> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = match m.role {
            MsgRole::User | MsgRole::Agent => "user",
            MsgRole::Tool => "user",
            MsgRole::System => "system",
        };
        let mut v = serde_json::json!({ "role": role, "content": m.content });
        if m.role == MsgRole::Agent {
            v["name"] = serde_json::Value::String(m.name.clone());
        }
        out.push(v);
    }
    out
}

impl LlmConfig {
    /// Resolve effective settings from an `AgentConfig`.
    pub fn from_agent_config(cfg: &crate::agent::config::AgentConfig) -> Self {
        Self {
            base_url: cfg.llm_base_url.clone(),
            api_key: cfg.llm_api_key.clone(),
            model: cfg.llm_model.clone(),
            temperature: cfg.llm_temperature,
            max_tokens: cfg.llm_max_tokens,
            timeout_secs: cfg.llm_timeout_secs,
            meter: cfg.meter.clone(),
        }
    }

    /// Attach an explicit meter (used by `AgentConfig::llm`).
    pub fn with_meter(mut self, meter: Arc<UsageMeter>) -> Self {
        self.meter = meter;
        self
    }
}

impl LlmConfig {
    /// Perform a single chat completion and return the assistant text.
    ///
    /// Rate limits are enforced (and token usage recorded) against `scope`, the
    /// current user/session namespace.
    pub async fn complete(
        &self,
        scope: &str,
        system: &str,
        messages: &[Message],
    ) -> Result<String, GenError> {
        self.meter.try_acquire(scope)?;
        let mut wire = Vec::new();
        if !system.is_empty() {
            wire.push(serde_json::json!({ "role": "system", "content": system }));
        }
        wire.extend(to_wire(messages));

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| GenError::Backend("agent-llm", format!("http client: {e}")))?;

        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": wire,
            "temperature": self.temperature,
        });
        if let Some(mt) = self.max_tokens {
            payload["max_tokens"] = serde_json::Value::Number(mt.into());
        }

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let req = client.post(&url).bearer_auth(&self.api_key).json(&payload);
        let resp = req.send().await.map_err(|e| {
            GenError::Backend("agent-llm", format!("request {url}: {e}"))
        })?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(GenError::Backend("agent-llm", format!("HTTP {}: {body}", status)));
        }
        let parsed: ChatResp = resp.json().await.map_err(|e| {
            GenError::Backend("agent-llm", format!("parse response: {e}"))
        })?;
        if let Some(u) = parsed.usage {
            self.meter.record(scope, u);
        }
        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| GenError::Backend("agent-llm", "empty content".to_string()))
    }

    /// Convenience: one-shot completion from a raw user prompt with only a system prompt.
    pub async fn chat(&self, scope: &str, system: &str, user: &str) -> Result<String, GenError> {
        let msgs = vec![Message {
            role: MsgRole::User,
            name: "user".into(),
            content: user.to_string(),
            meta: None,
        }];
        self.complete(scope, system, &msgs).await
    }
}
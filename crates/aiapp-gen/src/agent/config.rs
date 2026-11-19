//! Runtime configuration for the multi-agent engine. Read from environment
//! variables so it composes well with the existing `aiapp-gen` config.

use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::usage::{RateLimitConfig, UsageMeter};

/// Runtime settings for the orchestration engine.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    // --- LLM ---
    pub llm_base_url: String,
    pub llm_api_key: String,
    pub llm_model: String,
    pub llm_temperature: f32,
    pub llm_max_tokens: Option<u32>,
    pub llm_timeout_secs: u64,
    // --- Workspace / fencing ---
    /// Root directory in which the engine (and its agents) are allowed to
    /// operate. All file-tool paths are resolved and confined to this root.
    pub root_dir: PathBuf,
    /// Where sessions' conversation history is persisted.
    pub sessions_dir: PathBuf,
    /// Where long-term memory is persisted (JSON file).
    pub memory_file: PathBuf,
    /// Where skill packages are installed (imported `.zip`).
    pub skills_dir: PathBuf,
    // --- Web search ---
    /// If set, searches are routed to this API endpoint.
    pub web_search_api: Option<String>,
    /// Optional API key for the web-search endpoint.
    pub web_search_key: Option<String>,
    // --- Generation ---
    /// Skip LLM calls and use a deterministic mock when set. Useful for tests
    /// and offline demos (mirrors `AIAPP_BACKEND=mock`).
    pub mock: bool,
    /// Engine-wide usage meter + rate limiter (shared, atomic).
    pub meter: Arc<UsageMeter>,
}

impl AgentConfig {
    /// Defaults rooted under `./jobs/agent` (sibling of the engine's working dir).
    pub fn default_with_root(root: PathBuf) -> Self {
        Self {
            llm_base_url: "https://api.openai.com/v1".into(),
            llm_api_key: String::new(),
            llm_model: "gpt-4o-mini".into(),
            llm_temperature: 0.4,
            llm_max_tokens: None,
            llm_timeout_secs: 120,
            root_dir: root.clone(),
            sessions_dir: root.join("sessions"),
            memory_file: root.join("memory.json"),
            skills_dir: root.join("skills"),
            web_search_api: None,
            web_search_key: None,
            mock: false,
            meter: UsageMeter::new(RateLimitConfig::default()),
        }
    }
}

impl AgentConfig {
    /// Read configuration from environment variables. See docs table.
    ///
    /// | Env | Meaning | Default |
    /// |---|---|---|
    /// | `AGENT_LLM_BASE_URL` | OpenAI-compatible base URL | `https://api.openai.com/v1` |
    /// | `AGENT_LLM_API_KEY` | API key | empty |
    /// | `AGENT_LLM_MODEL` | model name | `gpt-4o-mini` |
    /// | `AGENT_LLM_TEMPERATURE` | sampling temperature | `0.4` |
    /// | `AGENT_LLM_MAX_TOKENS` | max output tokens | none |
    /// | `AGENT_LLM_TIMEOUT` | per-request timeout seconds | `120` |
    /// | `AGENT_ROOT` | root / fenccing dir | `./jobs/agent` |
    /// | `AGENT_SESSIONS_DIR` | sessions dir | `{root}/sessions` |
    /// | `AGENT_MEMORY_FILE` | memory file | `{root}/memory.json` |
    /// | `AGENT_SKILLS_DIR` | skills install dir | `{root}/skills` |
    /// | `AGENT_WEB_SEARCH_API` | search endpoint | none |
    /// | `AGENT_WEB_SEARCH_KEY` | search api key | none |
    /// | `AGENT_MOCK` | `1` to use deterministic mock | `0` |
    /// | `AGENT_LIMIT_GLOBAL_RPM` | max LLM requests/minute (all scopes) | none |
    /// | `AGENT_LIMIT_PER_SCOPE_RPM` | max LLM requests/minute per scope | none |
    /// | `AGENT_LIMIT_GLOBAL_DAILY_TOKENS` | max tokens/day (all scopes) | none |
    /// | `AGENT_LIMIT_PER_SCOPE_DAILY_TOKENS` | max tokens/day per scope | none |
    pub fn from_env() -> Self {
        let root = std::env::var("AGENT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./jobs/agent"));
        let sessions_dir = std::env::var("AGENT_SESSIONS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("sessions"));
        let memory_file = std::env::var("AGENT_MEMORY_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("memory.json"));
        let skills_dir = std::env::var("AGENT_SKILLS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("skills"));
        let mock = std::env::var("AGENT_MOCK").map(|v| v == "1").unwrap_or(false);
        let temperature = std::env::var("AGENT_LLM_TEMPERATURE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.4);
        let max_tokens = std::env::var("AGENT_LLM_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok());
        let timeout = std::env::var("AGENT_LLM_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120);
        let env_u64 = |k: &str| std::env::var(k).ok().and_then(|s| s.parse().ok());
        let meter = UsageMeter::new(RateLimitConfig {
            global_rpm: env_u64("AGENT_LIMIT_GLOBAL_RPM"),
            per_scope_rpm: env_u64("AGENT_LIMIT_PER_SCOPE_RPM"),
            global_daily_token_cap: env_u64("AGENT_LIMIT_GLOBAL_DAILY_TOKENS"),
            per_scope_daily_token_cap: env_u64("AGENT_LIMIT_PER_SCOPE_DAILY_TOKENS"),
        });
        Self {
            llm_base_url: std::env::var("AGENT_LLM_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
            llm_api_key: std::env::var("AGENT_LLM_API_KEY").unwrap_or_default(),
            llm_model: std::env::var("AGENT_LLM_MODEL")
                .unwrap_or_else(|_| "gpt-4o-mini".into()),
            llm_temperature: temperature,
            llm_max_tokens: max_tokens,
            llm_timeout_secs: timeout,
            root_dir: root.clone(),
            sessions_dir,
            memory_file,
            skills_dir,
            web_search_api: std::env::var("AGENT_WEB_SEARCH_API").ok(),
            web_search_key: std::env::var("AGENT_WEB_SEARCH_KEY").ok(),
            mock,
            meter,
        }
    }

    /// Build the shared LLM config from this agent config.
    pub fn llm(&self) -> crate::agent::llm::LlmConfig {
        crate::agent::llm::LlmConfig::from_agent_config(self).with_meter(self.meter.clone())
    }
}
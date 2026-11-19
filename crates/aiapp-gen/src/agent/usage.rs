//! Token accounting + rate limiting for the multi-agent engine.
//!
//! Every LLM call flows through `UsageMeter`:
//!  1. `try_acquire(scope)` — enforce requests-per-minute and daily-token budgets
//!     *before* dispatching the HTTP request.
//!  2. `record(scope, usage)` — accumulate the response's `usage` token counts.
//!
//! All checks and bookkeeping happen under a single mutex, so concurrent modes
//! (parallel fan-out / group chat) cannot race and overshoot a budget.

use crate::GenError;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_minute() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() / 60)
        .unwrap_or(0)
}

/// Calendar day used to roll daily budgets over at local midnight.
fn today() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

/// Token usage parsed from the LLM response `usage` field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl TokenUsage {
    pub fn total(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Aggregated counters, handy for stats / observability.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct UsageStats {
    pub calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

impl UsageStats {
    fn add(&mut self, u: TokenUsage) {
        self.calls += 1;
        self.prompt_tokens += u.prompt_tokens;
        self.completion_tokens += u.completion_tokens;
        self.total_tokens += u.total();
    }
}

/// Budgets enforced by the engine. Each is optional; `None` disables it.
#[derive(Debug, Clone, Copy, Default)]
pub struct RateLimitConfig {
    /// Max requests per minute across all scopes.
    pub global_rpm: Option<u64>,
    /// Max requests per minute for a single scope.
    pub per_scope_rpm: Option<u64>,
    /// Max tokens per calendar day across all scopes.
    pub global_daily_token_cap: Option<u64>,
    /// Max tokens per calendar day for a single scope.
    pub per_scope_daily_token_cap: Option<u64>,
}

#[derive(Debug, Default)]
struct Inner {
    global: UsageStats,
    per_scope: HashMap<String, UsageStats>,
    /// minute-window request counts: key -> (minute, count). `"*"` = global.
    rpm: HashMap<String, (u64, u64)>,
    /// daily token usage: scope key -> (date, tokens). `"*"` = global.
    daily: HashMap<String, (String, u64)>,
}

/// Shared, engine-wide usage meter + rate limiter.
#[derive(Debug)]
pub struct UsageMeter {
    limits: RateLimitConfig,
    inner: Mutex<Inner>,
}

impl UsageMeter {
    pub fn new(limits: RateLimitConfig) -> Arc<Self> {
        Arc::new(Self { limits, inner: Mutex::new(Inner::default()) })
    }

    pub fn limits(&self) -> RateLimitConfig {
        self.limits
    }

    /// Atomic pre-flight check + bookkeeping. Returns an error when any enabled
    /// budget would be exceeded, so the caller should NOT dispatch the request.
    pub fn try_acquire(&self, scope: &str) -> Result<(), GenError> {
        let minute = now_minute();
        let date = today();
        let mut g = self.inner.lock().unwrap();

        if let Some(rpm) = self.limits.global_rpm {
            if tick_rpm(&mut g.rpm, "*", minute) > rpm {
                return Err(deny("global requests per minute", rpm));
            }
        }
        if let Some(rpm) = self.limits.per_scope_rpm {
            if tick_rpm(&mut g.rpm, scope, minute) > rpm {
                return Err(deny(&format!("scope `{scope}` requests per minute"), rpm));
            }
        }
        if let Some(cap) = self.limits.global_daily_token_cap {
            if daily_tokens(&mut g.daily, "*", &date) >= cap {
                return Err(deny("global daily token budget", cap));
            }
        }
        if let Some(cap) = self.limits.per_scope_daily_token_cap {
            if daily_tokens(&mut g.daily, scope, &date) >= cap {
                return Err(deny(&format!("scope `{scope}` daily token budget"), cap));
            }
        }
        Ok(())
    }

    /// Accumulate an LLM response's token usage. Never fails.
    pub fn record(&self, scope: &str, usage: TokenUsage) {
        let date = today();
        let mut g = self.inner.lock().unwrap();
        g.global.add(usage);
        g.per_scope.entry(scope.to_string()).or_default().add(usage);
        for key in ["*", scope] {
            let e = g.daily.entry(key.to_string()).or_insert_with(|| (date.clone(), 0));
            if e.0 != date {
                e.0 = date.clone();
                e.1 = 0;
            }
            e.1 += usage.total();
        }
    }

    pub fn stats(&self) -> UsageStats {
        self.inner.lock().unwrap().global
    }

    pub fn scope_stats(&self, scope: &str) -> UsageStats {
        self.inner
            .lock()
            .unwrap()
            .per_scope
            .get(scope)
            .copied()
            .unwrap_or_default()
    }
}

/// Increment the per-minute counter for `key` and return its new value.
fn tick_rpm(map: &mut HashMap<String, (u64, u64)>, key: &str, minute: u64) -> u64 {
    let e = map.entry(key.to_string()).or_insert((minute, 0));
    if minute > e.0 {
        *e = (minute, 0);
    }
    e.1 += 1;
    e.1
}

/// Current daily token total for `key`, rolling over if the date changed.
fn daily_tokens(map: &mut HashMap<String, (String, u64)>, key: &str, date: &str) -> u64 {
    let e = map.entry(key.to_string()).or_insert_with(|| (date.to_string(), 0));
    if e.0 != *date {
        e.0 = date.to_string();
        e.1 = 0;
    }
    e.1
}

fn deny(what: &str, cap: u64) -> GenError {
    GenError::Backend(
        "agent-limit",
        format!("rate limit hit: {what} (cap={cap})"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meter(limits: RateLimitConfig) -> Arc<UsageMeter> {
        UsageMeter::new(limits)
    }

    #[test]
    fn rpm_blocks_the_call_that_exceeds_the_budget() {
        let m = meter(RateLimitConfig {
            global_rpm: Some(2),
            per_scope_rpm: None,
            global_daily_token_cap: None,
            per_scope_daily_token_cap: None,
        });
        assert!(m.try_acquire("u1").is_ok());
        assert!(m.try_acquire("u1").is_ok());
        assert!(m.try_acquire("u2").is_err(), "global RPM exhausted");
    }

    #[test]
    fn per_scope_rpm_isolates_users() {
        let m = meter(RateLimitConfig {
            global_rpm: Some(10),
            per_scope_rpm: Some(1),
            global_daily_token_cap: None,
            per_scope_daily_token_cap: None,
        });
        assert!(m.try_acquire("a").is_ok());
        assert!(m.try_acquire("a").is_err(), "scope a exceeded its RPM");
        assert!(m.try_acquire("b").is_ok(), "scope b unaffected");
    }

    #[test]
    fn records_and_accumulates_token_usage_per_scope() {
        let m = meter(RateLimitConfig::default());
        m.record("u1", TokenUsage { prompt_tokens: 100, completion_tokens: 50 });
        m.record("u1", TokenUsage { prompt_tokens: 30, completion_tokens: 20 });
        m.record("u2", TokenUsage { prompt_tokens: 5, completion_tokens: 5 });

        let g = m.stats();
        assert_eq!(g.calls, 3);
        assert_eq!(g.total_tokens, 100 + 50 + 30 + 20 + 5 + 5);

        let u1 = m.scope_stats("u1");
        assert_eq!(u1.calls, 2);
        assert_eq!(u1.total_tokens, 100 + 50 + 30 + 20);
    }

    #[test]
    fn daily_token_cap_blocks_once_budget_is_reached() {
        let m = meter(RateLimitConfig {
            global_rpm: None,
            per_scope_rpm: None,
            global_daily_token_cap: None,
            per_scope_daily_token_cap: Some(100),
        });
        m.record("u1", TokenUsage { prompt_tokens: 100, completion_tokens: 0 });
        assert!(m.try_acquire("u1").is_err(), "daily token budget already exhausted");
        assert!(m.try_acquire("u2").is_ok(), "other scope still within budget");
    }
}
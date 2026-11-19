//! Web tools: real-time web search and page-content fetching, over async HTTP.
//!
//! Search uses a configurable endpoint (`AGENT_WEB_SEARCH_API`); if none is
//! configured, `web_search` returns a guidance message. `fetch_url` is always
//! available and returns the (plain-text) body of the target page.

use super::ToolCtx;
use crate::GenError;
use serde_json::Value;

/// Search the web. `{query, n?}`.
pub async fn search(ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let a = args.as_object().cloned().unwrap_or_default();
    let query = a
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim();
    if query.is_empty() {
        return Err(GenError::Backend("web_search", "missing `query`".into()));
    }
    let n = a.get("n").and_then(|v| v.as_u64()).unwrap_or(5);

    let Some(endpoint) = &ctx.config.web_search_api else {
        return Err(GenError::Backend(
            "web_search",
            "AGENT_WEB_SEARCH_API is not configured; `fetch_url` is still available.".into(),
        ));
    };
    let client = reqwest::Client::new();
    let url = if endpoint.contains("{q}") {
        endpoint.replace("{q}", &urlencoding::encode(query))
    } else {
        format!("{endpoint}{}", urlencoding::encode(query))
    };
    let mut req = client.get(&url);
    if let Some(key) = &ctx.config.web_search_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.map_err(|e| {
        GenError::Backend("web_search", format!("request {url}: {e}"))
    })?;
    let text = resp.text().await.map_err(|e| GenError::Backend("web_search", e.to_string()))?;
    let mut items: Vec<String> = Vec::new();
    if let Ok(json) = serde_json::from_str::<Value>(&text) {
        items = extract_results(&json);
    }
    if items.is_empty() {
        // Fall back to trimmed raw text as the result.
        let trimmed = truncate_lines(&text, 40);
        items.push(trimmed);
    }
    items.truncate(n as usize);
    Ok(items.join("\n\n"))
}

fn extract_results(v: &Value) -> Vec<String> {
    // Handles common result shapes (Serper/You/Brave-ish and generic arrays):
    let mut out = Vec::new();
    extract_from_array(v.as_array(), &mut out);
    if out.is_empty() {
        for key in ["organic", "results", "web", "items", "data"] {
            if let Some(arr) = v.get(key).and_then(|x| x.as_array()) {
                extract_from_array(Some(arr), &mut out);
                if !out.is_empty() {
                    break;
                }
            }
        }
    }
    out
}

fn extract_from_array(arr: Option<&Vec<Value>>, out: &mut Vec<String>) {
    if let Some(arr) = arr {
        for it in arr {
            let title = it.get("title").and_then(|x| x.as_str()).unwrap_or("");
            let url = it
                .get("link")
                .or_else(|| it.get("url"))
                .and_then(|x| x.as_str())
                .unwrap_or("");
            let snippet = it.get("snippet").and_then(|x| x.as_str()).unwrap_or("");
            if !title.is_empty() || !url.is_empty() {
                out.push(format!("- {title}\n  {url}\n  {snippet}"));
            }
        }
    }
}

/// Fetch a URL and return its plain-text body. `{url}`.
pub async fn fetch(_ctx: &ToolCtx, args: Value) -> Result<String, GenError> {
    let a = args.as_object().cloned().unwrap_or_default();
    let url = a
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if url.is_empty() {
        return Err(GenError::Backend("fetch_url", "missing `url`".into()));
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(GenError::Backend("fetch_url", "url must be http(s)".into()));
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| GenError::Backend("fetch_url", e.to_string()))?;
    let resp = client
        .get(&url)
        .header("User-Agent", concat!("aiapp-agent/", env!("CARGO_PKG_VERSION")))
        .send()
        .await
        .map_err(|e| GenError::Backend("fetch_url", format!("{url}: {e}")))?;
    if !resp.status().is_success() {
        return Err(GenError::Backend("fetch_url", format!("HTTP {} {}", resp.status(), resp.url())));
    }
    let body = resp.text().await.map_err(|e| GenError::Backend("fetch_url", e.to_string()))?;
    Ok(strip_html(&body))
}

fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut skip = 0usize;
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            skip = 1;
            i += 1;
            continue;
        }
        if skip > 0 {
            skip += 1;
            if bytes[i] == b'>' {
                skip = 0;
            }
            i += 1;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    truncate_lines(&out, 200)
}

fn truncate_lines(s: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = s.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let total = lines.len();
    let mut out = lines.into_iter().take(max_lines).collect::<Vec<_>>().join("\n");
    if total > max_lines {
        out.push_str(&format!("\n... ({} more lines)", total - max_lines));
    }
    out
}
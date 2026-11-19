//! Local mock backend: no external AI dependency.
//!
//! The mock backend does **not** generate functional source code from a
//! natural-language description — that requires a real LLM.  Instead it returns
//! a clear error guiding the user to configure an AI backend (`openai` or
//! `pro`), so the UI never silently produces an app whose content is
//! unrelated to what the user asked for.

use crate::GenError;

/// Return an error explaining that AI generation is not configured.
pub fn generate(_desc: &str) -> Result<String, GenError> {
    Err(GenError::Backend(
        "mock",
        "AI 生成未启用。请配置 AIAPP_BACKEND=openai 并设置 AIAPP_OPENAI_API_KEY，\
         或选择一个内置模板（todo / pomodoro / image-filter 等）以使用离线模板。"
            .into(),
    ))
}

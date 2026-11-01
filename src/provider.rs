//! Config, the registered service handle, and the AI provider.

use std::sync::Arc;

use rustali_core::{AppContext, Provider, RegisterFuture, Registry, Result};
use serde::{Deserialize, Serialize};

use crate::{
    backend::{Backend, MockBackend},
    client::{Ai, SharedAi},
};

/// `config/ai.toml`.
///
/// ```toml
/// backend = "mock"                                   # mock | candle
/// model = "sentence-transformers/all-MiniLM-L6-v2"   # HF repo id (candle backend)
/// model_path = ""                                    # local dir; wins over `model` when set
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AiConfig {
    /// Backend: `mock` (default, deterministic, no model or network) or `candle` (local
    /// inference; requires the `candle` feature on `rustali-ai`).
    pub backend: String,
    /// Hugging Face repo id to download when no local path is given.
    pub model: String,
    /// Local directory holding `model.safetensors`, `tokenizer.json` and `config.json`.
    /// Takes precedence over `model` when non-empty, so inference needs no network.
    pub model_path: String,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            backend: "mock".into(),
            model: "sentence-transformers/all-MiniLM-L6-v2".into(),
            model_path: String::new(),
        }
    }
}

impl AiConfig {
    /// Section name inside the configuration tree.
    pub const SECTION: &'static str = "ai";
}

/// Registers an [`Ai`] handle from `config/ai.toml`.
#[derive(Debug, Default, Clone, Copy)]
pub struct AiProvider;

impl Provider for AiProvider {
    fn name(&self) -> &'static str {
        "ai"
    }

    fn register<'a>(&'a self, registry: &'a mut Registry<'_>) -> RegisterFuture<'a> {
        Box::pin(async move {
            let config: AiConfig = registry.config().section_or_default(AiConfig::SECTION)?;
            let backend = build(&config)?;
            registry.insert(Arc::new(Ai::new(backend)));
            Ok(())
        })
    }
}

#[cfg(not(feature = "candle"))]
fn build(config: &AiConfig) -> Result<Arc<dyn Backend>> {
    match config.backend.as_str() {
        "candle" => Err(rustali_core::Error::bad_request(
            "ai backend `candle` needs the `candle` feature on rustali-ai",
        )),
        _ => Ok(Arc::new(MockBackend::default())),
    }
}

#[cfg(feature = "candle")]
fn build(config: &AiConfig) -> Result<Arc<dyn Backend>> {
    match config.backend.as_str() {
        "candle" => Ok(Arc::new(crate::candle::CandleBackend::load(config)?)),
        _ => Ok(Arc::new(MockBackend::default())),
    }
}

/// `ctx.ai()` sugar on the application context.
pub trait AiContextExt {
    /// The registered AI handle, or an error when [`AiProvider`] was not added.
    ///
    /// # Errors
    /// Returns an error if no [`AiProvider`] registered a handle.
    fn ai(&self) -> Result<SharedAi>;
}

impl AiContextExt for AppContext {
    fn ai(&self) -> Result<SharedAi> {
        self.service::<SharedAi>().map(|arc| (*arc).clone())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{AiConfig, build};

    #[test]
    fn default_config_uses_mock() {
        let config = AiConfig::default();
        assert_eq!(config.backend, "mock");
        let backend = build(&config).map(|b| b.dimensions());
        assert!(matches!(backend, Ok(dims) if dims > 0));
    }

    #[test]
    #[cfg(not(feature = "candle"))]
    fn candle_backend_without_feature_errors() {
        let config = AiConfig {
            backend: "candle".into(),
            ..AiConfig::default()
        };
        // Map away the non-Debug Ok value so we can assert on the error message.
        let message = build(&config).map(|_| ()).map_err(|e| e.to_string());
        match message {
            Err(msg) => assert!(
                msg.to_lowercase().contains("candle"),
                "error should name the missing feature: {msg}"
            ),
            Ok(()) => panic!("expected an error without the candle feature"),
        }
    }
}

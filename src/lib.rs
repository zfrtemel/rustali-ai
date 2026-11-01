//! AI for the [Rustali](https://github.com/zfrtemel/rustali) web framework.
//!
//! Two capabilities behind one handle: [`Backend`] turns text into **embeddings**, and
//! [`Chat`] does **text generation**. Add [`AiProvider`] to your app and reach both from a
//! handler with `ctx.ai()`. This lets a Rust web app do AI **in process** — no Python service
//! alongside it.
//!
//! Chat and embeddings are independent — a handle may provide either or both. The traits are
//! provider-agnostic, so remote providers (`OpenAI`, Anthropic) slot in behind them later.
//!
//! Embeddings ship with two backends:
//!
//! - **`mock`** (default) — a deterministic, dependency-free [`MockBackend`] for tests and
//!   local development: no model, no network.
//! - **`candle`** (feature `candle`) — local inference with Hugging Face
//!   [Candle](https://github.com/huggingface/candle), running a sentence-transformer model
//!   (default `all-MiniLM-L6-v2`) on the CPU, in your own process.
//!
//! The trait is provider-agnostic, so remote providers (`OpenAI`, Anthropic) can slot in behind
//! the same API later.
//!
//! ```no_run
//! use rustali_ai::{AiProvider, AiContextExt, cosine_similarity};
//! # async fn demo(ctx: &rustali_core::AppContext) -> rustali_core::Result<()> {
//! let ai = ctx.ai()?;
//! let a = ai.embed("the cat sat on the mat").await?;
//! let b = ai.embed("a feline rested on the rug").await?;
//! let score = cosine_similarity(&a, &b); // ~1.0 means very similar
//! # let _ = (AiProvider, score);
//! # Ok(())
//! # }
//! ```

mod backend;
mod chat;
mod client;
mod provider;
mod vector_store;

#[cfg(feature = "candle")]
mod candle;

pub use backend::{Backend, EmbedFuture, MockBackend};
pub use chat::{Chat, ChatFuture, ChatMessage, MockChat, Role};
pub use client::{Ai, SharedAi, cosine_similarity};
pub use provider::{AiConfig, AiContextExt, AiProvider};
pub use vector_store::{SearchResult, VectorStore};

#[cfg(feature = "candle")]
pub use candle::CandleBackend;

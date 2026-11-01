//! The application-facing [`Ai`] handle and the [`cosine_similarity`] helper.

use std::sync::Arc;

use rustali_core::{Error, Result};

use crate::{
    backend::Backend,
    chat::{Chat, ChatMessage},
};

/// Shared AI handle, stored as a service and handed to handlers via `ctx.ai()`.
pub type SharedAi = Arc<Ai>;

/// The handle application code uses for AI.
///
/// Wraps an embedding [`Backend`] and, optionally, a [`Chat`] backend (local, a test mock, or
/// a future remote provider) behind a stable API, so swapping backends never changes call
/// sites. The two capabilities are independent: a handle may provide embeddings, chat, or
/// both.
#[derive(Clone)]
pub struct Ai {
    backend: Arc<dyn Backend>,
    chat: Option<Arc<dyn Chat>>,
}

impl Ai {
    /// Build a handle over an embedding `backend`, with no chat backend.
    #[must_use]
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            chat: None,
        }
    }

    /// Add a [`Chat`] backend, enabling [`chat`](Self::chat) / [`prompt`](Self::prompt).
    #[must_use]
    pub fn with_chat(mut self, chat: Arc<dyn Chat>) -> Self {
        self.chat = Some(chat);
        self
    }

    /// Embed a single string into a vector.
    ///
    /// # Errors
    /// Propagates any error from the underlying backend (model failure, network, …).
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.backend.embed_batch(&[text.to_owned()]).await?;
        Ok(out.pop().unwrap_or_default())
    }

    /// Embed many strings at once, returning one vector per input in order.
    ///
    /// Prefer this over calling [`embed`](Self::embed) in a loop: a backend can batch the
    /// work (one model forward pass, one network round-trip) instead of one per text.
    ///
    /// # Errors
    /// Propagates any error from the underlying backend.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.backend.embed_batch(texts).await
    }

    /// The dimensionality of vectors this handle's backend produces.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.backend.dimensions()
    }

    /// Whether a chat backend is configured.
    #[must_use]
    pub fn has_chat(&self) -> bool {
        self.chat.is_some()
    }

    /// Generate a reply to a full conversation.
    ///
    /// Build the `messages` with [`ChatMessage::system`] / [`ChatMessage::user`] /
    /// [`ChatMessage::assistant`]. For a one-shot question, [`prompt`](Self::prompt) is
    /// shorter.
    ///
    /// # Errors
    /// Returns an error if no chat backend is configured, or the backend fails.
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let chat = self.chat.as_ref().ok_or_else(|| {
            Error::internal(std::io::Error::other(
                "no chat backend configured on this Ai handle",
            ))
        })?;
        chat.generate(messages).await
    }

    /// Generate a reply to a single user message — the one-shot shortcut over
    /// [`chat`](Self::chat).
    ///
    /// # Errors
    /// Returns an error if no chat backend is configured, or the backend fails.
    pub async fn prompt(&self, text: &str) -> Result<String> {
        self.chat(&[ChatMessage::user(text)]).await
    }
}

impl std::fmt::Debug for Ai {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ai")
            .field("dimensions", &self.backend.dimensions())
            .field("has_chat", &self.chat.is_some())
            .finish()
    }
}

/// Cosine similarity of two vectors: `1.0` identical direction, `0.0` orthogonal, `-1.0`
/// opposite.
///
/// This is the usual way to ask "how similar are these two pieces of text?" once they are
/// embedded — rank candidates by their similarity to a query to find the closest matches
/// (semantic search, RAG retrieval).
///
/// Returns `0.0` when the vectors have different lengths, are empty, or either is all
/// zeros, rather than panicking or dividing by zero.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

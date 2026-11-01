//! The application-facing [`Ai`] handle and the [`cosine_similarity`] helper.

use std::sync::Arc;

use rustali_core::Result;

use crate::backend::Backend;

/// Shared AI handle, stored as a service and handed to handlers via `ctx.ai()`.
pub type SharedAi = Arc<Ai>;

/// The handle application code uses to produce embeddings.
///
/// Wraps a [`Backend`] (local Candle, a test mock, or a future remote provider) behind a
/// stable API, so swapping the backend never changes call sites.
#[derive(Clone)]
pub struct Ai {
    backend: Arc<dyn Backend>,
}

impl Ai {
    /// Build a handle over `backend`.
    #[must_use]
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
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
}

impl std::fmt::Debug for Ai {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ai")
            .field("dimensions", &self.backend.dimensions())
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

//! A small in-memory [`VectorStore`] for retrieval (the "R" in RAG).
//!
//! Store texts, then search by meaning: the store embeds what you add and, on a query, embeds
//! the query and returns the most similar entries by [`cosine_similarity`](crate::cosine_similarity).
//! It combines the embedding backend and similarity into the two calls a retrieval step needs —
//! `add` and `search` — so wiring up "give it my documents, find the relevant ones" is short.
//!
//! It holds everything in memory and scans linearly on search: ideal up to thousands of
//! entries. For larger corpora, reach for a dedicated vector database and keep this as the
//! reference shape.

use rustali_core::Result;

use crate::client::{Ai, cosine_similarity};

/// One stored entry: the original text, its embedding, and optional caller metadata.
#[derive(Debug, Clone)]
struct Entry {
    text: String,
    payload: Option<String>,
    vector: Vec<f32>,
}

/// A hit from [`VectorStore::search`], ranked by `score` (higher is more similar).
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The stored text.
    pub text: String,
    /// The caller metadata stored alongside it, if any (e.g. a source URL or document id).
    pub payload: Option<String>,
    /// Cosine similarity to the query, in `[-1.0, 1.0]`.
    pub score: f32,
}

/// An in-memory collection of embedded texts you can search by meaning.
#[derive(Debug, Default, Clone)]
pub struct VectorStore {
    entries: Vec<Entry>,
}

impl VectorStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Embed `text` with `ai` and store it.
    ///
    /// # Errors
    /// Propagates any error from the embedding backend.
    pub async fn add(&mut self, ai: &Ai, text: impl Into<String>) -> Result<()> {
        self.add_with(ai, text, None::<String>).await
    }

    /// Embed `text` and store it with caller `payload` (e.g. a source id) returned on search.
    ///
    /// # Errors
    /// Propagates any error from the embedding backend.
    pub async fn add_with(
        &mut self,
        ai: &Ai,
        text: impl Into<String>,
        payload: Option<impl Into<String>>,
    ) -> Result<()> {
        let text = text.into();
        let vector = ai.embed(&text).await?;
        self.entries.push(Entry {
            text,
            payload: payload.map(Into::into),
            vector,
        });
        Ok(())
    }

    /// Embed and store many texts in one batch (one model pass).
    ///
    /// # Errors
    /// Propagates any error from the embedding backend.
    pub async fn add_batch(&mut self, ai: &Ai, texts: &[String]) -> Result<()> {
        let vectors = ai.embed_batch(texts).await?;
        for (text, vector) in texts.iter().zip(vectors) {
            self.entries.push(Entry {
                text: text.clone(),
                payload: None,
                vector,
            });
        }
        Ok(())
    }

    /// Return the `top_k` entries most similar to `query`, best first.
    ///
    /// Embeds the query, scores every entry by cosine similarity, and returns the highest.
    /// Returns fewer than `top_k` when the store holds fewer entries.
    ///
    /// # Errors
    /// Propagates any error from the embedding backend.
    pub async fn search(&self, ai: &Ai, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        if self.entries.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let query_vec = ai.embed(query).await?;
        let mut scored: Vec<SearchResult> = self
            .entries
            .iter()
            .map(|e| SearchResult {
                text: e.text.clone(),
                payload: e.payload.clone(),
                score: cosine_similarity(&query_vec, &e.vector),
            })
            .collect();
        // Highest score first; total_cmp handles NaN without panicking.
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Remove every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

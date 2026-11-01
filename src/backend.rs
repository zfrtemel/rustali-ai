//! The [`Backend`] trait every embedding provider implements, and a deterministic
//! [`MockBackend`] for tests.

use std::{future::Future, pin::Pin};

use rustali_core::Result;

/// Future returned by [`Backend::embed_batch`].
pub type EmbedFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

/// An embedding backend: turns text into vectors.
///
/// The batch method is the primitive — a single embedding is a batch of one. The trait is
/// object-safe so a backend lives behind `Arc<dyn Backend>`, and it is deliberately
/// provider-agnostic: a future remote backend (`OpenAI`, Anthropic) implements the same
/// surface as the local Candle one.
pub trait Backend: Send + Sync + 'static {
    /// Embed every input string, returning one vector per input, in the same order.
    fn embed_batch<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a>;

    /// The dimensionality of the vectors this backend produces (e.g. 384 for
    /// all-MiniLM-L6-v2).
    fn dimensions(&self) -> usize;
}

/// A deterministic, dependency-free backend for tests and local development.
///
/// It needs no model and no network: each input maps to a fixed, L2-normalised vector
/// derived from the text, so the same text always yields the same vector and
/// [`cosine_similarity`](crate::cosine_similarity) of a text with itself is `1.0`.
#[derive(Debug, Clone, Copy)]
pub struct MockBackend {
    dims: usize,
}

impl MockBackend {
    /// A mock backend producing `dims`-dimensional vectors.
    #[must_use]
    pub const fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self { dims: 8 }
    }
}

impl Backend for MockBackend {
    fn embed_batch<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a> {
        let dims = self.dims;
        let out: Vec<Vec<f32>> = texts.iter().map(|t| deterministic_vec(t, dims)).collect();
        Box::pin(async move { Ok(out) })
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

/// Build a stable, L2-normalised pseudo-embedding from `text`.
///
/// Each component is seeded from an FNV-1a hash of the text mixed with the component index,
/// then the whole vector is normalised to unit length. Pure and deterministic — no RNG.
fn deterministic_vec(text: &str, dims: usize) -> Vec<f32> {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let base = text.bytes().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    });

    let mut raw: Vec<f32> = (0..dims)
        .map(|i| {
            let mixed = (base ^ (i as u64).wrapping_mul(FNV_PRIME)).wrapping_mul(FNV_PRIME);
            // Take 16 stable bits and map them to a float in [-1.0, 1.0) — a plain
            // u16 -> f32 conversion, no precision loss.
            let bits = u16::try_from((mixed >> 32) & 0xFFFF).unwrap_or(0);
            (f32::from(bits) / f32::from(u16::MAX)).mul_add(2.0, -1.0)
        })
        .collect();

    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut raw {
            *x /= norm;
        }
    }
    raw
}

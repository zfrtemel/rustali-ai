//! Unit tests that run with no model and no network (the `mock` backend).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use rustali_ai::{Ai, Backend, MockBackend, cosine_similarity};

#[test]
fn cosine_similarity_edges() {
    // identical -> 1.0
    let v = vec![1.0, 2.0, 3.0];
    assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);

    // orthogonal -> 0.0
    assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);

    // opposite -> -1.0
    assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);

    // mismatched length / empty / zero -> 0.0, never a panic
    assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
}

#[tokio::test]
async fn mock_embeddings_are_deterministic_and_normalized() {
    let ai = Ai::new(Arc::new(MockBackend::new(16)));
    assert_eq!(ai.dimensions(), 16);

    let a = ai.embed("hello world").await.unwrap();
    let b = ai.embed("hello world").await.unwrap();
    assert_eq!(a.len(), 16);
    assert_eq!(a, b, "same text -> same vector");

    // Deterministic + normalized => cosine with itself is 1.0.
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

    let c = ai.embed("something else entirely").await.unwrap();
    assert_ne!(a, c, "different text -> different vector");
}

#[tokio::test]
async fn embed_batch_preserves_order_and_count() {
    let ai = Ai::new(Arc::new(MockBackend::default()));
    let texts = vec!["one".to_owned(), "two".to_owned(), "three".to_owned()];
    let batch = ai.embed_batch(&texts).await.unwrap();

    assert_eq!(batch.len(), 3);
    // Each batched vector matches the single-embed of the same text, in order.
    for (i, text) in texts.iter().enumerate() {
        let single = ai.embed(text).await.unwrap();
        assert_eq!(batch[i], single);
    }
}

fn make(dims: usize) -> Ai {
    Ai::new(Arc::new(MockBackend::new(dims)))
}

/// A backend can be swapped behind the `Ai` handle without changing call sites.
#[tokio::test]
async fn ai_handle_is_backend_agnostic() {
    assert_eq!(make(4).embed("x").await.unwrap().len(), 4);
    assert_eq!(make(32).embed("x").await.unwrap().len(), 32);
}

/// The trait object is `Send + Sync` (required to live in `Arc<dyn Backend>`).
#[test]
fn backend_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arc<dyn Backend>>();
}

//! `VectorStore` retrieval tests over the deterministic `mock` backend.
//!
//! The mock embeds each text to a fixed vector, so an exact-text query scores 1.0 against its
//! stored copy and ranks first — enough to exercise add/search/ranking without a real model.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use rustali_ai::{Ai, MockBackend, VectorStore};

fn ai() -> Ai {
    Ai::new(Arc::new(MockBackend::new(32)))
}

#[tokio::test]
async fn add_and_search_ranks_exact_match_first() {
    let ai = ai();
    let mut store = VectorStore::new();
    assert!(store.is_empty());

    store.add(&ai, "the quick brown fox").await.unwrap();
    store.add(&ai, "a slow green turtle").await.unwrap();
    store.add(&ai, "rust is a systems language").await.unwrap();
    assert_eq!(store.len(), 3);

    let hits = store
        .search(&ai, "rust is a systems language", 2)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].text, "rust is a systems language");
    assert!((hits[0].score - 1.0).abs() < 1e-5, "exact match ~1.0");
    assert!(hits[0].score >= hits[1].score, "sorted best first");
}

#[tokio::test]
async fn search_respects_top_k_and_empty() {
    let ai = ai();
    let store = VectorStore::new();
    // Empty store -> no hits, no panic.
    assert!(store.search(&ai, "anything", 5).await.unwrap().is_empty());

    let mut store = VectorStore::new();
    let texts: Vec<String> = (0..5).map(|i| format!("doc number {i}")).collect();
    store.add_batch(&ai, &texts).await.unwrap();
    assert_eq!(store.len(), 5);

    assert_eq!(store.search(&ai, "doc number 2", 3).await.unwrap().len(), 3);
    assert_eq!(store.search(&ai, "doc number 2", 0).await.unwrap().len(), 0);
    // top_k larger than the store returns all entries.
    assert_eq!(
        store.search(&ai, "doc number 2", 99).await.unwrap().len(),
        5
    );
}

#[tokio::test]
async fn payload_is_returned_with_hits() {
    let ai = ai();
    let mut store = VectorStore::new();
    store
        .add_with(&ai, "chapter one", Some("book.pdf#1"))
        .await
        .unwrap();

    let hits = store.search(&ai, "chapter one", 1).await.unwrap();
    assert_eq!(hits[0].payload.as_deref(), Some("book.pdf#1"));
}

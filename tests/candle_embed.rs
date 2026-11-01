//! Real local-inference test: downloads all-MiniLM-L6-v2 and embeds sentences.
//!
//! Only compiled with `--features candle`, and marked `#[ignore]` because it downloads a
//! model (~90 MB) and is not suitable for CI. Run it manually:
//!
//! ```text
//! cargo test --features candle -- --ignored
//! ```

#![cfg(feature = "candle")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use rustali_ai::{Ai, AiConfig, CandleBackend, cosine_similarity};

#[tokio::test]
#[ignore = "downloads a model from the Hugging Face Hub; run with --ignored"]
async fn embeds_and_ranks_similarity() {
    let config = AiConfig {
        backend: "candle".into(),
        ..AiConfig::default()
    };
    let backend = CandleBackend::load(&config).expect("load MiniLM");
    let ai = Ai::new(Arc::new(backend));

    assert_eq!(ai.dimensions(), 384, "all-MiniLM-L6-v2 has 384 dims");

    let query = ai.embed("the cat sat on the mat").await.unwrap();
    let paraphrase = ai.embed("a feline rested on the rug").await.unwrap();
    let unrelated = ai
        .embed("quarterly revenue grew in the third fiscal period")
        .await
        .unwrap();

    assert_eq!(query.len(), 384);

    let sim_para = cosine_similarity(&query, &paraphrase);
    let sim_unrel = cosine_similarity(&query, &unrelated);

    assert!(
        sim_para > sim_unrel,
        "paraphrase ({sim_para}) should score higher than unrelated ({sim_unrel})"
    );
}

#[tokio::test]
#[ignore = "downloads a model from the Hugging Face Hub; run with --ignored"]
async fn batch_matches_single() {
    let config = AiConfig {
        backend: "candle".into(),
        ..AiConfig::default()
    };
    let ai = Ai::new(Arc::new(CandleBackend::load(&config).expect("load")));

    let texts = vec!["hello".to_owned(), "world".to_owned()];
    let batch = ai.embed_batch(&texts).await.unwrap();
    assert_eq!(batch.len(), 2);

    // A single embed of the same text should be very close to its batched vector.
    let single = ai.embed("hello").await.unwrap();
    assert!(cosine_similarity(&batch[0], &single) > 0.99);
}

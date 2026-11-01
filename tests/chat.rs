//! Chat trait tests over the deterministic `MockChat` backend — no model, no network.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use rustali_ai::{Ai, ChatMessage, MockBackend, MockChat};

fn ai_with_chat() -> Ai {
    Ai::new(Arc::new(MockBackend::default())).with_chat(Arc::new(MockChat))
}

#[tokio::test]
async fn prompt_returns_a_reply() {
    let ai = ai_with_chat();
    assert!(ai.has_chat());
    let reply = ai.prompt("hello there").await.unwrap();
    assert!(
        reply.contains("hello there"),
        "mock echoes the prompt: {reply}"
    );
}

#[tokio::test]
async fn chat_uses_the_last_user_message() {
    let ai = ai_with_chat();
    let messages = vec![
        ChatMessage::system("you are terse"),
        ChatMessage::user("first question"),
        ChatMessage::assistant("an answer"),
        ChatMessage::user("the real question"),
    ];
    let reply = ai.chat(&messages).await.unwrap();
    assert!(reply.contains("the real question"), "{reply}");
}

#[tokio::test]
async fn chat_without_backend_errors() {
    // Default handle has embeddings but no chat.
    let ai = Ai::new(Arc::new(MockBackend::default()));
    assert!(!ai.has_chat());
    let result = ai.prompt("hi").await.map_err(|e| e.to_string());
    match result {
        Err(msg) => assert!(msg.to_lowercase().contains("chat"), "{msg}"),
        Ok(reply) => panic!("expected an error, got: {reply}"),
    }
}

#[test]
fn chat_trait_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arc<dyn rustali_ai::Chat>>();
}

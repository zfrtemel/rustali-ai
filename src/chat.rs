//! The [`Chat`] trait for text generation, a [`ChatMessage`], and a deterministic
//! [`MockChat`] for tests.

use std::{future::Future, pin::Pin};

use rustali_core::Result;
use serde::{Deserialize, Serialize};

/// Future returned by [`Chat::generate`].
pub type ChatFuture<'a> = Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

/// Who authored a [`ChatMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions that steer the assistant (the "system prompt").
    System,
    /// A message from the end user.
    User,
    /// A previous reply from the assistant, for multi-turn context.
    Assistant,
}

/// One message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Who sent it.
    pub role: Role,
    /// The text.
    pub content: String,
}

impl ChatMessage {
    /// A system message (instructions for the assistant).
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    /// A user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// An assistant message (a prior reply).
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A text-generation backend: a conversation in, a reply out.
///
/// Object-safe so it lives behind `Arc<dyn Chat>`, and provider-agnostic: a local model, a
/// test mock, or a remote provider (Anthropic, `OpenAI`) all implement the same surface.
pub trait Chat: Send + Sync + 'static {
    /// Generate a reply to the conversation `messages`.
    fn generate<'a>(&'a self, messages: &'a [ChatMessage]) -> ChatFuture<'a>;
}

/// A deterministic, dependency-free chat backend for tests and local development.
///
/// It runs no model and needs no network: it replies with a fixed, readable summary of the
/// last user message, so tests can assert on its output.
#[derive(Debug, Default, Clone, Copy)]
pub struct MockChat;

impl Chat for MockChat {
    fn generate<'a>(&'a self, messages: &'a [ChatMessage]) -> ChatFuture<'a> {
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map_or_else(|| "(no user message)".to_owned(), |m| m.content.clone());
        let reply = format!("mock reply to: {last_user}");
        Box::pin(async move { Ok(reply) })
    }
}

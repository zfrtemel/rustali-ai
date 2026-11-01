# rustali-ai

AI for the [Rustali](https://github.com/zfrtemel/rustali) web framework — so a Rust web app
can do AI **in process**, no Python service alongside it.

Two capabilities behind one handle:

- **Embeddings** — turn text into vectors you can compare (semantic search, RAG).
- **Chat** — text generation (prompt in, reply out).

They are independent, and both sit behind provider-agnostic traits, so remote providers
(OpenAI, Anthropic) can slot in behind them later without changing your call sites.

## Backends

| Backend | Feature | What it does |
|---|---|---|
| `mock` (default) | — | Deterministic fake vectors. No model, no network. For tests and local dev. |
| `candle` | `candle` | Local inference with Hugging Face [Candle](https://github.com/huggingface/candle): runs a sentence-transformer (default `all-MiniLM-L6-v2`) on the CPU, in your process. |

`candle` is **opt-in** because it is a heavy dependency (long first compile). The default build
stays fast.

## Install

```toml
[dependencies]
rustali-ai = { git = "https://github.com/zfrtemel/rustali-ai" }
# local inference:
rustali-ai = { git = "https://github.com/zfrtemel/rustali-ai", features = ["candle"] }
```

## Use

Add the provider, then reach the handle from a request with `ctx.ai()`:

```rust
use rustali_ai::{AiProvider, AiContextExt, cosine_similarity};

// in your App builder:
// App::builder().provider(AiProvider).controllers().run().await

async fn similar(ctx: &rustali_core::AppContext) -> rustali_core::Result<()> {
    let ai = ctx.ai()?;

    let query = ai.embed("the cat sat on the mat").await?;
    let candidate = ai.embed("a feline rested on the rug").await?;

    let score = cosine_similarity(&query, &candidate); // ~1.0 = very similar
    println!("similarity: {score}");
    Ok(())
}
```

Embed many texts at once (one model pass instead of one per text):

```rust
let texts = vec!["first".to_owned(), "second".to_owned()];
let vectors = ai.embed_batch(&texts).await?; // Vec<Vec<f32>>, in order
```

### Chat (text generation)

```rust
use rustali_ai::ChatMessage;

// one-shot
let reply = ai.prompt("Write a haiku about Rust").await?;

// a full conversation
let reply = ai.chat(&[
    ChatMessage::system("You are a terse assistant."),
    ChatMessage::user("Summarize the borrow checker in one line."),
]).await?;
```

With the default `mock` backend, `prompt`/`chat` return a deterministic stub (handy for
tests). Real chat backends — a local model or a remote provider — slot in behind the same
`Chat` trait.

### "Find the most similar"

The common use: embed a query, embed your candidates, rank by cosine similarity.

```rust
let query = ai.embed(user_question).await?;
let mut ranked: Vec<_> = documents
    .iter()
    .zip(ai.embed_batch(&doc_texts).await?)
    .map(|(doc, vec)| (doc, cosine_similarity(&query, &vec)))
    .collect();
ranked.sort_by(|a, b| b.1.total_cmp(&a.1)); // best first
```

## Configuration

`config/ai.toml`:

```toml
backend = "candle"                                  # mock | candle
model = "sentence-transformers/all-MiniLM-L6-v2"    # HF repo id (downloaded + cached)
model_path = ""                                     # local dir; wins over `model` when set
```

With `model_path` pointing at a folder containing `model.safetensors`, `tokenizer.json` and
`config.json`, inference needs no network.

## Testing

```bash
cargo test                              # fast, mock backend, no network
cargo test --features candle            # unit tests; the download test is ignored
cargo test --features candle -- --ignored   # real: downloads MiniLM and embeds
```

## License

MIT OR Apache-2.0.

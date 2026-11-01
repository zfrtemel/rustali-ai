# rustali-ai

AI for the [Rustali](https://github.com/zfrtemel/rustali) web framework — so a Rust web app
can do AI **in process**, no Python service alongside it.

This first version does **embeddings**: it turns text into vectors you can compare, which is
the foundation of semantic search and retrieval (RAG). A single `Backend` trait keeps the API
stable; remote providers (OpenAI, Anthropic) can slot in behind it later.

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

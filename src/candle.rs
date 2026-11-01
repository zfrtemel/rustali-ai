//! Local embedding inference with Hugging Face Candle.
//!
//! Loads a BERT sentence-transformer (default `all-MiniLM-L6-v2`) once at startup and runs it
//! on the CPU, in process. The standard sentence-embedding recipe is used: a forward pass,
//! **masked mean pooling** over the token embeddings, then **L2 normalisation** so vectors are
//! unit length and [`cosine_similarity`](crate::cosine_similarity) behaves.

use std::{path::PathBuf, sync::Arc};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use rustali_core::{Error, Result};
use tokenizers::{PaddingParams, Tokenizer};

use crate::{
    backend::{Backend, EmbedFuture},
    provider::AiConfig,
};

/// Wrap any displayable error as an internal error.
///
/// Several dependencies here surface `Box<dyn Error + Send + Sync>` (tokenizers) or their own
/// error types; `Error::internal` needs a `Sized` error, so route everything through the
/// message instead. This keeps one conversion for candle, tokenizers and hf-hub alike.
fn internal(err: impl std::fmt::Display) -> Error {
    Error::Internal(err.to_string().into())
}

/// The loaded model and tokenizer, shared behind an `Arc` so inference holds no `&mut`.
struct Inner {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dims: usize,
}

/// Local embedding backend backed by Candle. Load it once via [`CandleBackend::load`].
pub struct CandleBackend {
    inner: Arc<Inner>,
}

impl CandleBackend {
    /// Load the model and tokenizer described by `config`.
    ///
    /// Uses `config.model_path` when set (no network), otherwise downloads the repo named by
    /// `config.model` from the Hugging Face Hub and caches it. This is expensive and happens
    /// once; keep the returned backend for the process's lifetime.
    ///
    /// # Errors
    /// Returns an error if the files cannot be fetched or read, or the model fails to load.
    pub fn load(config: &AiConfig) -> Result<Self> {
        let files = resolve_files(config)?;
        let device = Device::Cpu;

        let config_json = std::fs::read_to_string(&files.config).map_err(internal)?;
        let bert_config: Config = serde_json::from_str(&config_json).map_err(internal)?;
        let dims = bert_config.hidden_size;

        let weights = std::fs::read(&files.weights).map_err(internal)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)
            .map_err(internal)?;
        let model = BertModel::load(vb, &bert_config).map_err(internal)?;

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(internal)?;
        // Pad every row in a batch to the same length so tokens stack into one tensor.
        tokenizer.with_padding(Some(PaddingParams::default()));

        Ok(Self {
            inner: Arc::new(Inner {
                model,
                tokenizer,
                device,
                dims,
            }),
        })
    }
}

impl Backend for CandleBackend {
    fn embed_batch<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a> {
        let inner = Arc::clone(&self.inner);
        let texts = texts.to_vec();
        Box::pin(async move {
            // Candle's forward pass is synchronous CPU math; keep it off the async runtime.
            tokio::task::spawn_blocking(move || inner.run(&texts))
                .await
                .map_err(internal)?
        })
    }

    fn dimensions(&self) -> usize {
        self.inner.dims
    }
}

impl Inner {
    /// Tokenise, run the model, mean-pool with the attention mask, and L2-normalise.
    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(internal)?;

        let mut ids = Vec::with_capacity(encodings.len());
        let mut masks = Vec::with_capacity(encodings.len());
        for enc in &encodings {
            let id_row = Tensor::new(enc.get_ids(), &self.device).map_err(internal)?;
            let mask_row = Tensor::new(enc.get_attention_mask(), &self.device).map_err(internal)?;
            ids.push(id_row);
            masks.push(mask_row);
        }
        let input_ids = Tensor::stack(&ids, 0).map_err(internal)?;
        let attention_mask = Tensor::stack(&masks, 0).map_err(internal)?;
        let token_type_ids = input_ids.zeros_like().map_err(internal)?;

        let embeddings = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(internal)?;

        let pooled = mean_pool(&embeddings, &attention_mask)?;
        let normalized = l2_normalize(&pooled)?;
        normalized.to_vec2::<f32>().map_err(internal)
    }
}

/// Mean-pool token embeddings `[batch, seq, hidden]` down to `[batch, hidden]`, weighting by
/// the attention mask so padding tokens do not contribute.
fn mean_pool(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask
        .to_dtype(DType::F32)
        .map_err(internal)?
        .unsqueeze(2)
        .map_err(internal)?; // [batch, seq, 1]
    let masked = embeddings.broadcast_mul(&mask).map_err(internal)?;
    let summed = masked.sum(1).map_err(internal)?; // [batch, hidden]
    let counts = mask
        .sum(1)
        .map_err(internal)?
        .clamp(1e-9, f32::INFINITY)
        .map_err(internal)?; // [batch, 1]
    summed.broadcast_div(&counts).map_err(internal)
}

/// L2-normalise each row of `[batch, hidden]` to a unit vector.
fn l2_normalize(vectors: &Tensor) -> Result<Tensor> {
    let norm = vectors
        .sqr()
        .map_err(internal)?
        .sum_keepdim(1)
        .map_err(internal)?
        .sqrt()
        .map_err(internal)?
        .clamp(1e-12, f32::INFINITY)
        .map_err(internal)?;
    vectors.broadcast_div(&norm).map_err(internal)
}

/// The three files a BERT sentence-transformer needs.
struct ModelFiles {
    weights: PathBuf,
    tokenizer: PathBuf,
    config: PathBuf,
}

/// Resolve the model files: a local directory when `model_path` is set, else download the
/// repo from the Hugging Face Hub (cached on disk by `hf-hub`).
fn resolve_files(config: &AiConfig) -> Result<ModelFiles> {
    if config.model_path.is_empty() {
        download_from_hub(&config.model)
    } else {
        let dir = PathBuf::from(&config.model_path);
        Ok(ModelFiles {
            weights: dir.join("model.safetensors"),
            tokenizer: dir.join("tokenizer.json"),
            config: dir.join("config.json"),
        })
    }
}

/// Download the three files of a `owner/name` repo from the Hub (cached on disk).
fn download_from_hub(repo_id: &str) -> Result<ModelFiles> {
    let (owner, name) = repo_id.split_once('/').ok_or_else(|| {
        Error::bad_request(format!("model repo id must be `owner/name`: {repo_id}"))
    })?;

    let client = hf_hub::HFClientSync::new().map_err(internal)?;
    let fetch = |file: &str| -> Result<PathBuf> {
        client
            .model(owner, name)
            .download_file()
            .filename(file)
            .send()
            .map_err(internal)
    };
    Ok(ModelFiles {
        weights: fetch("model.safetensors")?,
        tokenizer: fetch("tokenizer.json")?,
        config: fetch("config.json")?,
    })
}

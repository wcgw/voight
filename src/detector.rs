//! The "vanguard" AI-text classifier: ModernBERT-large fine-tuned to output a
//! single P(AI) logit.
//! <https://huggingface.co/ShantanuT01/vanguard-ai-text-detector>

use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use candle_transformers::models::modernbert;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use tokenizers::{Tokenizer, TruncationParams};

use crate::probability::Probability;

const REPO: &str = "ShantanuT01/vanguard-ai-text-detector";
/// Pinned so that a new upload to the repository can't silently change scores.
const REVISION: &str = "823061be63b90f2b42f64ac1e1f82772e872533b";
/// Inputs are truncated to this many tokens; windows from
/// [`crate::text::segments`] stay well below it.
const MAX_TOKENS: usize = 1024;

/// Paths to the files the detector is loaded from.
#[derive(Debug)]
pub struct ModelFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: PathBuf,
}

impl ModelFiles {
    /// The files in `dir`, laid out as in the model repository.
    pub fn in_dir(dir: &Path) -> Self {
        Self {
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
            weights: dir.join("model.safetensors"),
        }
    }

    /// The pinned model revision from the Hugging Face cache, downloading it
    /// (about 1.5 GB) the first time.
    pub fn fetch() -> Result<Self> {
        let pinned = Repo::with_revision(REPO.to_owned(), RepoType::Model, REVISION.to_owned());
        let cache = Cache::from_env();
        if cache
            .repo(pinned.clone())
            .get("model.safetensors")
            .is_none()
            && io::stderr().is_terminal()
        {
            eprintln!("Downloading the detector model (about 1.5 GB, first run only)…");
        }
        let api = ApiBuilder::from_cache(cache)
            .with_progress(true)
            .build()
            .context("failed to set up the Hugging Face client")?;
        let repo = api.repo(pinned);
        let get = |name: &str| {
            repo.get(name)
                .with_context(|| format!("failed to fetch {name} from {REPO}"))
        };
        Ok(Self {
            config: get("config.json")?,
            tokenizer: get("tokenizer.json")?,
            weights: get("model.safetensors")?,
        })
    }
}

/// The loaded classifier.
pub struct Detector {
    tokenizer: Tokenizer,
    backbone: modernbert::ModernBert,
    dense: Linear,
    norm: LayerNorm,
    classifier: Linear,
    device: Device,
}

impl Detector {
    pub fn load(files: &ModelFiles, device: Device) -> Result<Self> {
        let config: modernbert::Config = serde_json::from_slice(&read(&files.config)?)
            .with_context(|| format!("failed to parse {}", files.config.display()))?;

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(anyhow::Error::from_boxed)
            .with_context(|| format!("failed to load {}", files.tokenizer.display()))?;
        // The repository's tokenizer.json pads every input to 512 tokens; the
        // mean pooling in `score` would average the padding in with the text.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(anyhow::Error::from_boxed)?;

        // Read into memory rather than mmap: it avoids `unsafe`, and the
        // weights are copied into tensors either way.
        let vb = VarBuilder::from_buffered_safetensors(read(&files.weights)?, DType::F32, &device)?;
        let hidden = config.hidden_size;
        // candle's `ModernBertForSequenceClassification` ends in a softmax,
        // which is always 1.0 for this single-logit head, so the head is
        // assembled here: dense, GELU, LayerNorm, then the classifier.
        Ok(Self {
            backbone: modernbert::ModernBert::load(vb.clone(), &config)?,
            dense: Linear::new(vb.get((hidden, hidden), "head.dense.weight")?, None),
            norm: LayerNorm::new_no_bias(
                vb.get(hidden, "head.norm.weight")?,
                config.layer_norm_eps,
            ),
            classifier: Linear::new(
                vb.get((1, hidden), "classifier.weight")?,
                Some(vb.get(1, "classifier.bias")?),
            ),
            tokenizer,
            device,
        })
    }

    /// The probability that `text` was written by an LLM.
    pub fn score(&self, text: &str) -> Result<Probability> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::from_boxed)?;
        let ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let mask = Tensor::ones(ids.dims(), DType::U32, &self.device)?;
        let hidden = self.backbone.forward(&ids, &mask)?;
        // Mean pooling over every token: there is no padding to mask out.
        let pooled = hidden.mean(1)?;
        let x = self.dense.forward(&pooled)?.gelu_erf()?;
        let logit = self.classifier.forward(&self.norm.forward(&x)?)?;
        let logit = logit.flatten_all()?.get(0)?.to_scalar::<f32>()?;
        Ok(Probability::from_logit(f64::from(logit)))
    }
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

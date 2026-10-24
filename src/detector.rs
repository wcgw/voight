//! The AI-text classifiers: "gradient" (DeBERTa-v3-large, the default) and
//! "vanguard" (ModernBERT-large), fine-tuned by the same author on the same
//! data to output a single P(AI) logit.

use std::fmt;
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use tokenizers::{PostProcessor, Tokenizer, TruncationParams};

use crate::candle::core::{DType, Device, Module, Tensor};
use crate::candle::nn::{LayerNorm, Linear, VarBuilder};
use crate::candle::transformers::models::{debertav2, modernbert};
use crate::fingerprint;
use crate::probability::Probability;
use crate::report::Report;
use crate::text::Segment;

/// A detector model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[non_exhaustive]
pub enum Model {
    /// DeBERTa-v3-large, <https://huggingface.co/ShantanuT01/gradient-ai-text-detector>;
    /// reads 512 tokens at a time, so longer windows are split
    Gradient,
    /// ModernBERT-large, <https://huggingface.co/ShantanuT01/vanguard-ai-text-detector>
    Vanguard,
}

impl Model {
    fn repo(self) -> &'static str {
        match self {
            Self::Vanguard => "ShantanuT01/vanguard-ai-text-detector",
            Self::Gradient => "ShantanuT01/gradient-ai-text-detector",
        }
    }

    /// Pinned so that a new upload to the repository can't silently change
    /// scores.
    fn revision(self) -> &'static str {
        match self {
            Self::Vanguard => "823061be63b90f2b42f64ac1e1f82772e872533b",
            Self::Gradient => "c2e8b6df87f8a211cbffb713fa9873a0c3a9713f",
        }
    }

    /// The most tokens the model reads at once, special tokens included:
    /// gradient's is the length it was trained on.
    fn max_tokens(self) -> usize {
        match self {
            Self::Vanguard => 1024,
            Self::Gradient => 512,
        }
    }

    fn download_size(self) -> &'static str {
        match self {
            Self::Vanguard => "about 1.5 GB",
            Self::Gradient => "about 1.7 GB",
        }
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Vanguard => "vanguard",
            Self::Gradient => "gradient",
        })
    }
}

/// Paths to the files a detector is loaded from.
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

    /// The pinned revision of `model` from the Hugging Face cache,
    /// downloading it the first time, with a notice and a progress bar on
    /// stderr when it is a terminal.
    pub fn fetch(model: Model) -> Result<Self> {
        let pinned = Repo::with_revision(
            model.repo().to_owned(),
            RepoType::Model,
            model.revision().to_owned(),
        );
        let cache = Cache::from_env();
        if cache
            .repo(pinned.clone())
            .get("model.safetensors")
            .is_none()
            && io::stderr().is_terminal()
        {
            eprintln!(
                "Downloading the {model} detector model ({}, first run only)…",
                model.download_size()
            );
        }
        let api = ApiBuilder::from_cache(cache)
            .with_progress(true)
            .build()
            .context("failed to set up the Hugging Face client")?;
        let repo = api.repo(pinned);
        let get = |name: &str| {
            repo.get(name)
                .with_context(|| format!("failed to fetch {name} from {}", model.repo()))
        };
        Ok(Self {
            config: get("config.json")?,
            tokenizer: get("tokenizer.json")?,
            weights: get("model.safetensors")?,
        })
    }
}

/// A loaded classifier.
///
/// [`Self::report`] scores a whole text; [`Self::split_to_fit`] and
/// [`Report::new`] do the same in steps, for callers that report progress
/// as each window is scored.
pub struct Detector {
    tokenizer: Tokenizer,
    /// The same tokenizer without truncation, for measuring text.
    counter: Tokenizer,
    /// The most tokens of text the model reads, special tokens aside.
    token_limit: usize,
    network: Network,
    device: Device,
}

enum Network {
    ModernBert {
        backbone: modernbert::ModernBert,
        dense: Linear,
        norm: LayerNorm,
        classifier: Linear,
    },
    DeBerta(Box<debertav2::DebertaV2SeqClassificationModel>),
}

impl fmt::Debug for Detector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Detector")
            .field("token_limit", &self.token_limit)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Detector {
    /// Loads `model`'s weights from `files` onto `device`. Fails if a file
    /// can't be read or doesn't hold `model`.
    pub fn load(model: Model, files: &ModelFiles, device: Device) -> Result<Self> {
        let config = read(&files.config)?;
        let parse_error = || format!("failed to parse {}", files.config.display());

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(anyhow::Error::from_boxed)
            .with_context(|| format!("failed to load {}", files.tokenizer.display()))?;
        // Both repositories' tokenizer.json pad every input to 512 tokens;
        // the padding would be pooled (vanguard) or attended to (gradient)
        // along with the text.
        tokenizer.with_padding(None);
        let counter = tokenizer.clone();
        let specials = tokenizer
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(false));
        let token_limit = model.max_tokens().saturating_sub(specials);
        // Text longer than the limit should have been split with
        // `split_to_fit`; this keeps any that wasn't within what the model
        // can read.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: model.max_tokens(),
                ..Default::default()
            }))
            .map_err(anyhow::Error::from_boxed)?;

        // Read into memory rather than mmap: it avoids `unsafe`, and the
        // weights are copied into tensors either way.
        let vb = VarBuilder::from_buffered_safetensors(read(&files.weights)?, DType::F32, &device)?;
        let network = match model {
            Model::Vanguard => {
                let config: modernbert::Config =
                    serde_json::from_slice(&config).with_context(parse_error)?;
                let hidden = config.hidden_size;
                // candle's `ModernBertForSequenceClassification` ends in a
                // softmax, which is always 1.0 for this single-logit head, so
                // the head is assembled here: dense, GELU, LayerNorm, then the
                // classifier.
                Network::ModernBert {
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
                }
            }
            Model::Gradient => {
                let config: debertav2::Config =
                    serde_json::from_slice(&config).with_context(parse_error)?;
                // Returns the raw logit: the CLS pooler, then the classifier.
                Network::DeBerta(Box::new(debertav2::DebertaV2SeqClassificationModel::load(
                    vb.pp("deberta"),
                    &config,
                    None,
                )?))
            }
        };
        Ok(Self {
            tokenizer,
            counter,
            token_limit,
            network,
            device,
        })
    }

    /// Scores `text`'s windows of prose (see [`crate::segments`]), split to
    /// fit the model. Fails if `text` holds no prose or the model fails.
    pub fn report<'a>(&self, text: &'a str) -> Result<Report<'a>> {
        let segments = self.split_to_fit(crate::segments(text))?;
        Report::new(text, segments, |text| self.score(text))
    }

    /// `segments` with each window the model can't read whole split into
    /// pieces it can, at sentence boundaries where possible. Each piece is
    /// then scored as a window of its own.
    pub fn split_to_fit<'a>(&self, segments: Vec<Segment<'a>>) -> Result<Vec<Segment<'a>>> {
        let mut fitted = Vec::with_capacity(segments.len());
        for segment in segments {
            let lines = segment.lines.clone();
            let pieces = segment
                .split_to_fit(self.token_limit, |text| self.tokens(text))
                .with_context(|| format!("failed to measure lines {lines:?}"))?;
            fitted.extend(pieces);
        }
        Ok(fitted)
    }

    /// The number of tokens in `text`, as [`Self::score`] reads it, not
    /// counting the special tokens the model adds around it.
    fn tokens(&self, text: &str) -> Result<usize> {
        let encoding = self
            .counter
            .encode(&*fingerprint::normalize(text), false)
            .map_err(anyhow::Error::from_boxed)?;
        Ok(encoding.len())
    }

    /// The probability that `text` was written by an LLM, read with any
    /// disguise from detectors undone (see [`fingerprint::normalize`]). Reads
    /// only as many tokens as the model can; [`Self::split_to_fit`] splits
    /// longer windows.
    pub fn score(&self, text: &str) -> Result<Probability> {
        let encoding = self
            .tokenizer
            .encode(&*fingerprint::normalize(text), true)
            .map_err(anyhow::Error::from_boxed)?;
        let ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let logit = match &self.network {
            Network::ModernBert {
                backbone,
                dense,
                norm,
                classifier,
            } => {
                let mask = Tensor::ones(ids.dims(), DType::U32, &self.device)?;
                let hidden = backbone.forward(&ids, &mask)?;
                // Mean pooling over every token: there is no padding to mask out.
                let pooled = hidden.mean(1)?;
                let x = dense.forward(&pooled)?.gelu_erf()?;
                classifier.forward(&norm.forward(&x)?)?
            }
            // Without a mask every token is attended to: there is no padding.
            Network::DeBerta(model) => model.forward(&ids, None, None)?,
        };
        let logit = logit.flatten_all()?.get(0)?.to_scalar::<f32>()?;
        Ok(Probability::from_logit(f64::from(logit)))
    }
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

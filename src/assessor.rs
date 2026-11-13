//! voight at work beside a user interface: a thread of its own that loads the model, then assesses the texts it's
//! given, one at a time, in order, saying how it goes through a callback. ghd and muttrs both ran one of these; it
//! lives here so they share it, with nothing of either's interface in it.

use std::ops::Range;
use std::sync::mpsc;

use crate::{Detector, Device, Model, ModelFiles};

/// The fewest words of prose worth assessing: on fewer, the estimate means little.
pub const MIN_WORDS: usize = 50;

/// Where the model runs. A GPU that can't be opened falls back to the CPU, saying why in [`Event::Loaded`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Processor {
    Cpu,
    /// A Mac's GPU.
    Metal,
    /// An NVIDIA GPU.
    Cuda,
    /// Any other GPU, through wgpu: Vulkan on Linux. Only with voight's `wgpu` feature.
    Wgpu,
}

/// An assessment of a text: the share of it an LLM wrote, overall and in each window scored.
#[derive(Debug, Clone, PartialEq)]
pub struct Assessment {
    /// Between 0 and 1: the windows' shares, weighted by their words.
    pub share: f64,
    /// In the order they come in the text.
    pub windows: Vec<Window>,
}

/// A window of prose scored on its own.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// Where it is in the text, in bytes.
    pub range: Range<usize>,
    /// The share of it an LLM wrote, between 0 and 1.
    pub share: f64,
}

/// What the [`Assessor`] has to say, about texts it knows by `K`.
#[derive(Debug, Clone, PartialEq)]
pub enum Event<K> {
    /// The model is loaded. `fallback` says why it runs on the CPU rather than where it was asked to, if it does.
    Loaded { fallback: Option<String> },
    /// The model couldn't be loaded: there'll be no assessments.
    Failed(String),
    /// The assessment asked for under `key`; `None` for a text with fewer than [`MIN_WORDS`] words of prose.
    Assessed {
        key: K,
        assessment: Result<Option<Assessment>, String>,
    },
}

/// voight on a thread of its own, assessing the texts it's given under keys of type `K`.
#[derive(Debug)]
pub struct Assessor<K> {
    texts: mpsc::Sender<(K, String)>,
}

impl<K: Send + 'static> Assessor<K> {
    /// Fetches `model`'s files, then starts loading it on a thread of its own, to run on `processor`; `on_event` hears
    /// how loading went and every assessment, on that thread. Fetching downloads the model the first time (about
    /// 1.7 GB), showing progress on stderr, and blocks meanwhile: call it before a terminal UI takes the screen.
    ///
    /// # Errors
    ///
    /// When the model can't be fetched.
    pub fn start(
        model: Model,
        processor: Processor,
        on_event: impl Fn(Event<K>) + Send + 'static,
    ) -> anyhow::Result<Self> {
        let files = ModelFiles::fetch(model)?;
        let (texts, queued) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("voight".to_owned())
            .spawn(move || work(model, &files, processor, &queued, &on_event));
        if let Err(error) = spawned {
            anyhow::bail!("couldn't start a thread for voight: {error}");
        }
        Ok(Self { texts })
    }

    /// Asks for the assessment of `text`, to come as an [`Event::Assessed`] under `key`, after those asked for before.
    pub fn assess(&self, key: K, text: String) {
        // Once the thread has stopped, its failure has been said: the text goes unassessed.
        let _ = self.texts.send((key, text));
    }
}

/// The thread: loads `model` from `files` onto `processor`, then assesses the texts `queued` until the sender is
/// dropped, telling `on_event` how it goes.
fn work<K>(
    model: Model,
    files: &ModelFiles,
    processor: Processor,
    queued: &mpsc::Receiver<(K, String)>,
    on_event: &impl Fn(Event<K>),
) {
    let (device, fallback) = match processor {
        Processor::Cpu => (Device::Cpu, None),
        Processor::Metal => opened(Device::new_metal(0), "Metal"),
        Processor::Cuda => opened(Device::new_cuda(0), "CUDA"),
        #[cfg(feature = "wgpu")]
        Processor::Wgpu => opened(Device::new_wgpu(0), "wgpu"),
        #[cfg(not(feature = "wgpu"))]
        Processor::Wgpu => (
            Device::Cpu,
            Some("voight was built without wgpu, so it runs on the CPU".to_owned()),
        ),
    };
    let detector = match Detector::load(model, files, device) {
        Ok(detector) => detector,
        Err(error) => return on_event(Event::Failed(format!("{error:#}"))),
    };
    on_event(Event::Loaded { fallback });
    for (key, text) in queued {
        let assessment = assess(&detector, &text).map_err(|error| format!("{error:#}"));
        on_event(Event::Assessed { key, assessment });
    }
}

/// The device `opening` opened, or the CPU, with why, if it failed.
fn opened(opening: Result<Device, impl std::fmt::Display>, name: &str) -> (Device, Option<String>) {
    match opening {
        Ok(device) => (device, None),
        Err(error) => (
            Device::Cpu,
            Some(format!(
                "couldn't use {name} ({error}), so voight runs on the CPU"
            )),
        ),
    }
}

/// The assessment of `text` by `detector`; `None` if it has fewer than [`MIN_WORDS`] words of prose.
fn assess(detector: &Detector, text: &str) -> anyhow::Result<Option<Assessment>> {
    if !worth_assessing(text) {
        return Ok(None);
    }
    let report = detector.report(text)?;
    Ok(Some(Assessment {
        share: report.overall().value(),
        windows: report
            .windows()
            .map(|(window, llm)| Window {
                range: window.range.clone(),
                share: llm.value(),
            })
            .collect(),
    }))
}

/// Whether `text` has at least [`MIN_WORDS`] words of prose, as voight counts them: code doesn't count.
#[must_use]
pub fn worth_assessing(text: &str) -> bool {
    crate::segments(text)
        .iter()
        .map(|segment| segment.words)
        .sum::<usize>()
        >= MIN_WORDS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_few_words_or_code_alone_are_not_worth_assessing() {
        let prose = "word ".repeat(MIN_WORDS);
        let few = "word ".repeat(MIN_WORDS - 1);
        let code = format!("```\n{}\n```\n", "code ".repeat(MIN_WORDS * 2));

        assert!(worth_assessing(&prose));
        assert!(!worth_assessing(&few));
        assert!(!worth_assessing(&code));
    }

    #[test]
    fn an_assessor_can_be_moved_to_another_thread() {
        fn sent<T: Send>() {}
        sent::<Assessor<String>>();
        sent::<Event<String>>();
    }
}

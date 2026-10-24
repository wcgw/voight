//! Estimates how much of a text was written by an LLM.
//!
//! The text is split into windows of prose (paragraphs grouped into about
//! 100–400 words, fenced code skipped). Each window is scored by a classifier
//! fine-tuned to tell human from LLM-written text (DeBERTa "gradient" by
//! default, or ModernBERT "vanguard"), and the overall figure is the
//! word-weighted mean of the window scores. Ways of hiding text from
//! detectors are undone before scoring; [`Fingerprints`] says which were
//! found, and points at residue that chatbot interfaces leave in text.
//!
//! No detector is reliable enough to be used as evidence of authorship.
//!
//! ```no_run
//! use voight::{Detector, Device, Fingerprints, Model, ModelFiles};
//!
//! let text = std::fs::read_to_string("essay.md")?;
//! let model = Model::Gradient;
//! let detector = Detector::load(model, &ModelFiles::fetch(model)?, Device::Cpu)?;
//! let report = detector.report(&text)?;
//! println!("{}", report.overall());
//! for note in Fingerprints::find(report.windows().map(|(window, _)| window)).notes() {
//!     eprintln!("{note}");
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! The `cli` feature, on by default, builds the `voight` command; turn it
//! off to embed the library without clap. `cuda`, `metal` and `wgpu` let
//! [`Device`] run the model on a GPU.

mod candle;
mod detector;
mod fingerprint;
mod heatmap;
mod probability;
mod report;
mod text;

pub use crate::candle::core::Device;

pub use crate::detector::{Detector, Model, ModelFiles};
pub use crate::fingerprint::{Fingerprints, normalize};
pub use crate::heatmap::Heatmap;
pub use crate::probability::Probability;
pub use crate::report::Report;
pub use crate::text::{Segment, segments};

#[cfg(test)]
mod tests {
    /// So that a server, say, can score texts from several threads.
    #[test]
    fn a_detector_can_be_shared_between_threads() {
        fn shared<T: Send + Sync>() {}
        shared::<crate::Detector>();
    }
}

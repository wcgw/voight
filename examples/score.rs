//! Scores a file with the voight library, as another program embedding it
//! would:
//!
//! ```sh
//! cargo run --release --example score -- essay.md
//! ```

use std::env;
use std::fs;

use anyhow::{Context, Result};
use voight::{Detector, Device, Fingerprints, Model, ModelFiles};

fn main() -> Result<()> {
    let path = env::args().nth(1).context("usage: score FILE")?;
    let text = fs::read_to_string(&path).with_context(|| format!("failed to read {path}"))?;

    // The first run downloads the model into the Hugging Face cache;
    // `ModelFiles::in_dir` loads one from a directory instead.
    let model = Model::Gradient;
    let files = ModelFiles::fetch(model)?;
    // With the `cuda` or `metal` feature, `Device::new_cuda(0)` or
    // `Device::new_metal(0)` run the model on a GPU.
    let detector = Detector::load(model, &files, Device::Cpu)?;

    let report = detector.report(&text)?;
    for (window, llm) in report.windows() {
        println!(
            "lines {}-{}: {llm} of {} words",
            window.lines.start(),
            window.lines.end(),
            window.words
        );
    }
    println!("overall: {}", report.overall());

    // What was undone before scoring, and chatbot residue, with its line.
    let fingerprints = Fingerprints::find(report.windows().map(|(window, _)| window));
    for note in fingerprints.notes() {
        eprintln!("note: {note}");
    }
    Ok(())
}

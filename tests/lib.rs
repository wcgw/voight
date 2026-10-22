use std::fs;
use std::path::Path;

use voight::{Detector, Device, Model, ModelFiles};

/// Written by an LLM (Claude) for this test suite.
const LLM_STYLED: &str = "tests/fixtures/llm-styled.md";
/// Mark Twain, *Life on the Mississippi* (1883), chapter I; public domain.
const HUMAN: &str = "tests/fixtures/human.txt";

/// The default model, from `VOIGHT_MODEL_DIR` when it's set, else the
/// Hugging Face cache, on the CPU.
fn detector() -> Detector {
    let model = Model::Gradient;
    let files = match std::env::var_os("VOIGHT_MODEL_DIR") {
        Some(dir) => ModelFiles::in_dir(Path::new(&dir)),
        None => ModelFiles::fetch(model).expect("fetch the model"),
    };
    Detector::load(model, &files, Device::Cpu).expect("load the model")
}

fn fixture(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).expect("read fixture")
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn a_detector_reports_on_a_whole_text() {
    let detector = detector();
    let llm = fixture(LLM_STYLED);
    let human = fixture(HUMAN);
    assert!(detector.report(&llm).unwrap().overall().value() > 0.5);
    assert!(detector.report(&human).unwrap().overall().value() < 0.5);
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn text_without_prose_has_no_report() {
    let err = detector().report("```\nfn main() {}\n```\n").unwrap_err();
    assert_eq!(err.to_string(), "no words to score");
}

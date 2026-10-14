//! `voight` estimates how much of a text was written by an LLM.
//!
//! The text is split into windows of prose (paragraphs grouped into about
//! 100–400 words, fenced code skipped). Each window is scored by a ModernBERT
//! classifier fine-tuned to tell human from LLM-written text, and the overall
//! figure is the word-weighted mean of the window scores.
//!
//! No detector is reliable enough to be used as evidence of authorship.

mod detector;
mod heatmap;
mod input;
mod probability;
mod progress;
mod report;
mod text;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, ensure};
use candle_core::Device;
use clap::Parser;
use indicatif::ProgressBar;

use crate::detector::{Detector, ModelFiles};
use crate::heatmap::Heatmap;
use crate::input::{Input, Source, Text};
use crate::probability::Probability;
use crate::report::Report;
use crate::text::Segment;

/// Estimate how much of a text was written by an LLM.
///
/// Prints the estimated share of the text as a percentage, or one line per
/// file when given several files or a directory. The first run downloads the
/// detector model (about 1.5 GB) into the Hugging Face cache. No detector is
/// reliable enough to be used as evidence of authorship.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Files or directories to scan; reads stdin when none is given or for
    /// `-`. A directory is scanned for the files directly inside it.
    #[arg(value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Show the score of each window of the text.
    #[arg(short, long)]
    verbose: bool,

    /// Print the text itself with each window shaded from green (human) to
    /// red (LLM-written), using 24-bit terminal colours; page it with
    /// `less -R`.
    #[arg(long, conflicts_with = "verbose")]
    heatmap: bool,

    /// Load the model from DIR (config.json, tokenizer.json,
    /// model.safetensors) instead of the Hugging Face cache.
    #[arg(long, value_name = "DIR")]
    model_dir: Option<PathBuf>,

    /// Run on the CPU even when a GPU backend (CUDA, Metal) is available.
    #[arg(long)]
    cpu: bool,
}

/// How results are printed.
#[derive(Debug, Clone, Copy)]
enum Output {
    /// The overall percentage (one line per file in a batch).
    Percent,
    /// The score of each window (`--verbose`).
    Breakdown,
    /// The text shaded by score (`--heatmap`).
    Heatmap,
}

impl Output {
    fn from_args(args: &Args) -> Self {
        // clap rejects --verbose together with --heatmap.
        if args.heatmap {
            Self::Heatmap
        } else if args.verbose {
            Self::Breakdown
        } else {
            Self::Percent
        }
    }
}

fn main() -> Result<ExitCode> {
    let args = Args::parse();
    let output = Output::from_args(&args);
    match Input::from_paths(&args.paths)? {
        Input::Single(source) => single(&args, output, &source),
        Input::Batch(files) => batch(&args, output, &files),
    }
}

/// Scores one text and prints the result in the `output` form.
fn single(args: &Args, output: Output, source: &Source) -> Result<ExitCode> {
    let text = source.read()?;
    // Checked before loading the model, which takes a while.
    let segments = text::segments(&text);
    ensure!(!segments.is_empty(), "no prose to score in input");
    let detector = load_detector(args)?;

    let scoring = progress::scoring(segments.len());
    let report = Report::new(&text, segments, |text| score(&detector, &scoring, text))?;
    // Cleared before printing so the bar doesn't interleave with the result.
    scoring.finish_and_clear();
    match output {
        Output::Percent => println!("{}", report.overall()),
        Output::Breakdown => print!("{report}"),
        Output::Heatmap => print!("{}", Heatmap::new(&report)),
    }
    Ok(ExitCode::SUCCESS)
}

/// What to do with one file of a batch.
enum Plan<'a> {
    Score {
        text: &'a str,
        segments: Vec<Segment<'a>>,
    },
    Skip(&'static str),
    Fail(&'a anyhow::Error),
}

impl<'a> Plan<'a> {
    fn new(text: &'a Result<Text>) -> Self {
        match text {
            Err(err) => Self::Fail(err),
            Ok(Text::NotUtf8) => Self::Skip("not UTF-8 text"),
            Ok(Text::Utf8(text)) => {
                let segments = text::segments(text);
                if segments.is_empty() {
                    Self::Skip("no prose")
                } else {
                    Self::Score { text, segments }
                }
            }
        }
    }
}

/// Scores each file and prints its line as soon as it's done. Files that
/// can't be read or scored are reported on stderr and fail the exit status
/// once the rest are done; files without prose are listed as skipped.
fn batch(args: &Args, output: Output, files: &[PathBuf]) -> Result<ExitCode> {
    // Everything is read up front: the progress bar needs the total number
    // of windows, and the model only loads if some file has prose.
    let texts: Vec<Result<Text>> = files.iter().map(|path| Text::read(path)).collect();
    let plans: Vec<Plan<'_>> = texts.iter().map(Plan::new).collect();
    let windows = plans
        .iter()
        .map(|plan| match plan {
            Plan::Score { segments, .. } => segments.len(),
            Plan::Skip(_) | Plan::Fail(_) => 0,
        })
        .sum();

    let scoring = progress::scoring(windows);
    let mut detector = None;
    let mut failed = false;
    for (path, plan) in files.iter().zip(plans) {
        let name = path.file_name().map_or(path.as_os_str(), |name| name);
        scoring.set_message(name.to_string_lossy().into_owned());
        match plan {
            Plan::Score { text, segments } => {
                let detector = match &mut detector {
                    Some(detector) => detector,
                    // The bar steps aside while the loading spinner runs.
                    slot => slot.insert(scoring.suspend(|| load_detector(args))?),
                };
                match Report::new(text, segments, |text| score(detector, &scoring, text)) {
                    Ok(report) => scoring.suspend(|| print_scored(output, path, &report)),
                    Err(err) => {
                        failed = true;
                        scoring.suspend(|| eprintln!("voight: {}: {err:#}", path.display()));
                    }
                }
            }
            Plan::Skip(reason) => scoring.suspend(|| print_skipped(output, path, reason)),
            Plan::Fail(err) => {
                failed = true;
                scoring.suspend(|| eprintln!("voight: {err:#}"));
            }
        }
    }
    scoring.finish_and_clear();
    Ok(if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

/// Prints a batch file's result. The multi-line forms start with the path
/// and end with a blank line.
fn print_scored(output: Output, path: &Path, report: &Report<'_>) {
    match output {
        Output::Percent => println!("{:>4}  {}", report.overall(), path.display()),
        Output::Breakdown => println!("{}\n{report}", path.display()),
        Output::Heatmap => println!("{}\n{}", path.display(), Heatmap::new(report)),
    }
}

fn print_skipped(output: Output, path: &Path, reason: &str) {
    match output {
        Output::Percent => println!("{:>4}  {}  ({reason})", "-", path.display()),
        Output::Breakdown | Output::Heatmap => {
            println!("{}\nskipped: {reason}\n", path.display());
        }
    }
}

/// Scores `text`, advancing `bar` by one window.
fn score(detector: &Detector, bar: &ProgressBar, text: &str) -> Result<Probability> {
    let score = detector.score(text);
    bar.inc(1);
    score
}

/// Loads the detector onto the best available device, with a spinner.
fn load_detector(args: &Args) -> Result<Detector> {
    let files = match &args.model_dir {
        Some(dir) => ModelFiles::in_dir(dir),
        None => ModelFiles::fetch()?,
    };
    let device = if args.cpu { Device::Cpu } else { gpu_or_cpu() };
    let loading = progress::spinner(format!("Loading the detector onto {}", describe(&device)));
    let detector = Detector::load(&files, device).context("failed to load the detector")?;
    loading.finish_and_clear();
    Ok(detector)
}

/// The GPU backend compiled in (CUDA, else Metal), or the CPU. candle's
/// `*_is_available` only say whether a backend was compiled in, so a GPU
/// that fails to open falls back to the CPU with a warning.
fn gpu_or_cpu() -> Device {
    let gpu = if candle_core::utils::cuda_is_available() {
        ("CUDA", Device::new_cuda(0))
    } else if candle_core::utils::metal_is_available() {
        ("Metal", Device::new_metal(0))
    } else {
        return Device::Cpu;
    };
    match gpu {
        (_, Ok(device)) => device,
        (name, Err(err)) => {
            eprintln!("voight: {name} unavailable, using the CPU: {err}");
            Device::Cpu
        }
    }
}

fn describe(device: &Device) -> &'static str {
    if device.is_cuda() {
        "the GPU (CUDA)"
    } else if device.is_metal() {
        "the GPU (Metal)"
    } else {
        "the CPU"
    }
}

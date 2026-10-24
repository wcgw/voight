# voight

voight (inpired by the
[Voight-Kampff machine from Blade Runner](https://en.wikipedia.org/wiki/Blade_Runner#Voight-Kampff_machine)) 
estimates how much of a text was written by an LLM.

Splits text into windows of prose (~100–400 words, split smaller when too long
for the model; fenced code skipped). Each window scored by a DeBERTa classifier
(default) or a ModernBERT one (`--model vanguard`), both fine-tuned on human vs
LLM-written text. Overall figure = word-weighted mean of window scores.

Common ways of hiding text from detectors are undone before scoring:
invisible characters (zero-width spaces, soft hyphens, …) removed, look-alike
letters from other scripts inside Latin words (Cyrillic `а` for `a`) replaced,
fullwidth, styled and other compatibility forms of letters and unusual spaces
made plain. Other text reaches the model unchanged.

**No detector reliable enough to use as evidence of authorship.**

## Install

```
cargo install --path . --locked
```

First run downloads the model (~1.7 GB; vanguard ~1.5 GB) into the Hugging Face cache.

## Usage

```
voight [OPTIONS] [PATH]...
```

Reads stdin when no path given (or `-`). Directory scans files directly inside
it (not recursive).

### Options

RTFM, aka `voight --help`

### Examples

```sh
# Single file
voight essay.md

# Stdin
cat report.txt | voight

# Batch directory
voight docs/

# Window breakdown
voight --verbose essay.md

# Heatmap
voight --heatmap essay.md | less -R

# The other model
voight --model vanguard essay.md
```

## Output

Single file: one percentage line.

```
73%
```

Batch: one line per file, score aligned.

```
 12%  tests/fixtures/human.txt
 87%  tests/fixtures/llm-styled.md
```

Files without scorable prose are listed as skipped. Unreadable files printed to
stderr; exit status non-zero.

Notes on stderr (prefixed with the path in a batch) say what was undone before
scoring, and point at chatbot residue: text that chatbot interfaces leave in
copied answers (`utm_source=chatgpt.com` links, `contentReference[oaicite:…]`,
`turn0search…` citations, "As an AI language model").

```
voight: 140 invisible characters inside or between words, removed before scoring
voight: chatbot residue on line 14: utm_source=chatgpt.com (3 times)
```

Copy-paste and mojibake leave stray invisible characters in human text too.

## Library

Scoring is a library too. Depend on it with `default-features = false` to
leave out the command (and clap).

```rust
use voight::{Detector, Device, Model, ModelFiles};

let model = Model::Gradient;
let detector = Detector::load(model, &ModelFiles::fetch(model)?, Device::Cpu)?;
let report = detector.report(&text)?;
println!("{}", report.overall());
```

Whole program, with window scores and notes, in `examples/score.rs`:

```sh
cargo run --release --example score -- essay.md
```

## GPU

- **Apple silicon**: Metal by default (already compiled in).
- **NVIDIA**: build with `--features cuda` (needs CUDA toolkit).
- **Other Mac**: build with `--features metal`.
- **Any other GPU** (Intel, AMD; Vulkan on Linux): build with `--features wgpu`.
  Experimental: candle's wgpu backend is unreleased, so this builds on
  [a fork](https://github.com/huggingface/candle/pull/3379).
- Falls back to CPU with a warning if GPU fails to open.

## Tests

```sh
# Fast (no model needed)
cargo test

# Full (needs model)
VOIGHT_MODEL_DIR=/path/to/model cargo test -- --include-ignored
```

# voight

voight (inpired by the
[Voight-Kampff machine from Blade Runner](https://en.wikipedia.org/wiki/Blade_Runner#Voight-Kampff_machine)) 
estimates how much of a text was written by an LLM.

Splits text into windows of prose (~100–400 words, fenced code skipped). Each
window scored by a DeBERTa classifier (default) or a ModernBERT one
(`--model vanguard`), both fine-tuned on human vs LLM-written text. Overall
figure = word-weighted mean of window scores.

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

## GPU

- **Apple silicon**: Metal by default (already compiled in).
- **NVIDIA**: build with `--features cuda` (needs CUDA toolkit).
- **Other Mac**: build with `--features metal`.
- Falls back to CPU with a warning if GPU fails to open.

## Tests

```sh
# Fast (no model needed)
cargo test

# Full (needs model)
VOIGHT_MODEL_DIR=/path/to/model cargo test -- --include-ignored
```

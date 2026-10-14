use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// Written by an LLM (Claude) for this test suite.
const LLM_STYLED: &str = "tests/fixtures/llm-styled.md";
/// Mark Twain, *Life on the Mississippi* (1883), chapter I; public domain.
const HUMAN: &str = "tests/fixtures/human.txt";

/// Runs voight with `args` and `stdin`. When `VOIGHT_MODEL_DIR` is set and
/// `args` name neither a model nor a model dir, the default model is loaded
/// from there instead of the Hugging Face cache.
fn voight(args: &[&str], stdin: &str) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_voight"));
    if let Some(dir) = std::env::var_os("VOIGHT_MODEL_DIR")
        && !args.contains(&"--model-dir")
        && !args.contains(&"--model")
    {
        command.arg("--model-dir").arg(dir);
    }
    let mut child = command
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn voight");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(stdin.as_bytes())
        .expect("write stdin");
    child.wait_with_output().expect("wait for voight")
}

fn percent(output: &Output) -> u32 {
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .trim_end()
        .strip_suffix('%')
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("not a percentage: {stdout:?}"))
}

fn stderr(output: &Output) -> String {
    assert!(!output.status.success(), "{output:?}");
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn missing_file_fails() {
    let err = stderr(&voight(&["does/not/exist.txt"], ""));
    assert!(err.contains("failed to read does/not/exist.txt"), "{err}");
}

#[test]
fn input_without_prose_fails_before_loading_the_model() {
    for input in ["  \n\n", "```\nfn main() {}\n```\n", "42\n\n✅\n"] {
        let err = stderr(&voight(&["--model-dir", "does/not/exist"], input));
        assert!(err.contains("no prose to score"), "{input:?}: {err}");
    }
}

#[test]
fn missing_model_dir_fails() {
    let err = stderr(&voight(&["--model-dir", "does/not/exist"], "Some prose."));
    assert!(err.contains("failed to load the detector"), "{err}");
    assert!(err.contains("does/not/exist/config.json"), "{err}");
}

#[test]
fn model_dir_is_loaded_as_the_named_model() {
    let args = ["--model", "vanguard", "--model-dir", "does/not/exist"];
    let err = stderr(&voight(&args, "Some prose."));
    assert!(
        err.contains("failed to load the detector (vanguard)"),
        "{err}"
    );
    assert!(err.contains("does/not/exist/config.json"), "{err}");
}

#[test]
fn unknown_model_is_rejected() {
    let err = stderr(&voight(&["--model", "nope"], "Some prose."));
    assert!(err.contains("invalid value 'nope'"), "{err}");
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn scores_llm_text_high_and_human_text_low() {
    assert!(percent(&voight(&[LLM_STYLED], "")) > 50);
    assert!(percent(&voight(&[HUMAN], "")) < 50);
}

#[test]
#[ignore = "needs the 1.5 GB vanguard model"]
fn vanguard_scores_llm_text_high_and_human_text_low() {
    assert!(percent(&voight(&["--model", "vanguard", LLM_STYLED], "")) > 50);
    assert!(percent(&voight(&["--model", "vanguard", HUMAN], "")) < 50);
}

/// A single window of prose, 198 words long, that takes up about 790 of
/// gradient's tokens and 760 of vanguard's.
fn token_heavy_window() -> String {
    "Install it under /usr/lib64/libfoo-3.so and set \
     FOO_HOME=~/.config/foo-bar/qux.toml before running fooctl --sync. "
        .repeat(18)
}

/// The number of windows in a `--verbose` breakdown.
fn breakdown_rows(output: &Output) -> usize {
    assert!(output.status.success(), "{output:?}");
    stdout(output).lines().count() - 2 // the header and the total
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn windows_longer_than_the_model_reads_are_split_and_shaded_apart() {
    let text = token_heavy_window();
    assert_eq!(breakdown_rows(&voight(&["--verbose"], &text)), 2);
    let heatmap = voight(&["--heatmap"], &text);
    assert!(heatmap.status.success(), "{heatmap:?}");
    let heatmap = stdout(&heatmap);
    let (body, _footer) = heatmap.rsplit_once("\n\n").expect("a footer");
    assert_eq!(body.matches("\x1b[30;48;2;").count(), 2, "{body:?}");
}

#[test]
#[ignore = "needs the 1.5 GB vanguard model"]
fn vanguard_reads_the_same_window_whole() {
    let output = voight(&["--verbose", "--model", "vanguard"], &token_heavy_window());
    assert_eq!(breakdown_rows(&output), 1);
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn mixed_text_scores_between_its_parts() {
    let llm = std::fs::read_to_string(LLM_STYLED).expect("fixture");
    let human = std::fs::read_to_string(HUMAN).expect("fixture");
    let mixed = percent(&voight(&[], &format!("{llm}\n\n{human}")));
    let (llm, human) = (
        percent(&voight(&[LLM_STYLED], "")),
        percent(&voight(&[HUMAN], "")),
    );
    assert!(human < mixed && mixed < llm, "{human} {mixed} {llm}");
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn reads_stdin_when_no_file_or_dash() {
    let text = std::fs::read_to_string(LLM_STYLED).expect("fixture");
    let from_file = percent(&voight(&[LLM_STYLED], ""));
    assert_eq!(percent(&voight(&[], &text)), from_file);
    assert_eq!(percent(&voight(&["-"], &text)), from_file);
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn verbose_shows_each_window_and_the_total() {
    let llm = std::fs::read_to_string(LLM_STYLED).expect("fixture");
    let human = std::fs::read_to_string(HUMAN).expect("fixture");
    let output = voight(&["--verbose"], &format!("{llm}\n\n{human}"));
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<_> = stdout.lines().collect();
    assert!(lines[0].starts_with("lines"), "{stdout}");
    assert!(lines.last().unwrap().starts_with("overall"), "{stdout}");
    assert!(lines.len() >= 4, "expected a window per fixture: {stdout}");
}

/// An empty directory under cargo's per-target scratch space.
fn scratch_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir); // left over from an earlier run, if any
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// `(score, path)` for each line of batch output, the score without `%`.
fn batch_lines(output: &Output) -> Vec<(String, String)> {
    stdout(output)
        .lines()
        .map(|line| {
            let (score, rest) = line.trim_start().split_once("  ").expect("score  path");
            (score.trim_end_matches('%').to_owned(), rest.to_owned())
        })
        .collect()
}

#[test]
fn batch_skips_files_without_prose_without_loading_the_model() {
    let dir = scratch_dir("batch-skips");
    fs::write(dir.join("numbers.txt"), "42\n\n✅\n").unwrap();
    fs::write(dir.join("image.bin"), [0xff, 0xd8, 0xff, 0x00]).unwrap();
    fs::write(dir.join(".hidden.txt"), "Hidden prose is ignored.").unwrap();
    fs::create_dir(dir.join("nested")).unwrap();
    fs::write(dir.join("nested/inner.txt"), "Nested prose is ignored.").unwrap();

    let output = voight(
        &["--model-dir", "does/not/exist", dir.to_str().unwrap()],
        "",
    );
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        stdout(&output),
        format!(
            "   -  {}  (not UTF-8 text)\n   -  {}  (no prose)\n",
            dir.join("image.bin").display(),
            dir.join("numbers.txt").display()
        )
    );
}

#[test]
fn batch_reports_unreadable_files_and_fails() {
    let output = voight(
        &["--model-dir", "does/not/exist", "no/a.txt", "no/b.txt"],
        "",
    );
    let err = stderr(&output);
    assert!(err.contains("failed to read no/a.txt"), "{err}");
    assert!(err.contains("failed to read no/b.txt"), "{err}");
    assert_eq!(stdout(&output), "");
}

#[test]
fn stdin_cannot_join_a_batch() {
    let err = stderr(&voight(&["-", HUMAN], ""));
    assert!(err.contains("can't be combined"), "{err}");
}

#[test]
fn empty_directory_is_an_error() {
    let dir = scratch_dir("batch-empty");
    let err = stderr(&voight(&[dir.to_str().unwrap()], ""));
    assert!(err.contains("no files to scan"), "{err}");
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn batch_scores_each_file_of_a_directory_in_order() {
    let output = voight(&["tests/fixtures"], "");
    assert!(output.status.success(), "{output:?}");
    let lines = batch_lines(&output);
    assert_eq!(
        lines
            .iter()
            .map(|(_, path)| path.as_str())
            .collect::<Vec<_>>(),
        ["tests/fixtures/human.txt", "tests/fixtures/llm-styled.md"]
    );
    let score = |i: usize| lines[i].0.parse::<u32>().expect("a percentage");
    assert!(score(0) < 50 && score(1) > 50, "{lines:?}");
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn batch_keeps_going_past_failures() {
    let output = voight(&[HUMAN, "does/not/exist.txt", LLM_STYLED], "");
    let err = stderr(&output);
    assert!(err.contains("failed to read does/not/exist.txt"), "{err}");
    let paths: Vec<_> = batch_lines(&output)
        .into_iter()
        .map(|(_, path)| path)
        .collect();
    assert_eq!(paths, [HUMAN, LLM_STYLED]);
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn verbose_batch_breaks_down_each_file() {
    let output = voight(&["--verbose", HUMAN, LLM_STYLED], "");
    assert!(output.status.success(), "{output:?}");
    let out = stdout(&output);
    let blocks: Vec<_> = out.trim_end().split("\n\n").collect();
    assert_eq!(blocks.len(), 2, "{out}");
    for (block, path) in blocks.iter().zip([HUMAN, LLM_STYLED]) {
        let lines: Vec<_> = block.lines().collect();
        assert_eq!(lines[0], path, "{out}");
        assert!(lines[1].starts_with("lines"), "{out}");
        assert!(lines.last().unwrap().starts_with("overall"), "{out}");
    }
}

/// `output` without terminal colour codes.
fn plain(output: &str) -> String {
    let mut out = String::new();
    let mut rest = output;
    while let Some(start) = rest.find("\x1b[") {
        out.push_str(&rest[..start]);
        let end = rest[start..].find('m').expect("terminated escape");
        rest = &rest[start + end + 1..];
    }
    out + rest
}

#[test]
fn heatmap_and_verbose_are_exclusive() {
    let err = stderr(&voight(&["--heatmap", "--verbose", HUMAN], ""));
    assert!(err.contains("cannot be used with"), "{err}");
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn heatmap_prints_the_input_shaded_then_the_overall_score() {
    let output = voight(&["--heatmap", LLM_STYLED], "");
    assert!(output.status.success(), "{output:?}");
    let rendered = stdout(&output);
    assert!(
        rendered.contains("\x1b[30;48;2;"),
        "no colours: {rendered:?}"
    );
    let input = fs::read_to_string(LLM_STYLED).unwrap();
    let plain = plain(&rendered);
    let footer = plain
        .strip_prefix(&input)
        .expect("input reproduced verbatim");
    let overall = percent(&voight(&[LLM_STYLED], ""));
    assert!(
        footer.starts_with(&format!("\n{overall}% overall")),
        "{footer:?}"
    );
}

#[test]
#[ignore = "needs the 1.7 GB detector model"]
fn heatmap_batch_shows_each_file_under_its_path() {
    let output = voight(&["--heatmap", HUMAN, LLM_STYLED], "");
    assert!(output.status.success(), "{output:?}");
    let plain = plain(&stdout(&output));
    for path in [HUMAN, LLM_STYLED] {
        let input = fs::read_to_string(path).unwrap();
        assert!(
            plain.contains(&format!("{path}\n{input}\n")),
            "{path}: {plain}"
        );
    }
}

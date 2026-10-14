//! Resolving command-line paths into the texts to score.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

/// What the command line asks to scan.
#[derive(Debug, PartialEq)]
pub enum Input {
    /// One text, reported on its own.
    Single(Source),
    /// Several files, reported one line each.
    Batch(Vec<PathBuf>),
}

/// Where a single text comes from.
#[derive(Debug, PartialEq)]
pub enum Source {
    Stdin,
    File(PathBuf),
}

/// The contents of a file in a batch.
#[derive(Debug, PartialEq)]
pub enum Text {
    Utf8(String),
    /// Not a text file (or not UTF-8 text); skipped rather than failed.
    NotUtf8,
}

impl Input {
    /// No paths or `-` read stdin, and a single file is a single text.
    /// Anything else is a batch: directories expand to the non-hidden files
    /// directly inside them, sorted by name.
    pub fn from_paths(paths: &[PathBuf]) -> Result<Self> {
        let stdin = Path::new("-");
        match paths {
            [] => Ok(Self::Single(Source::Stdin)),
            [path] if path == stdin => Ok(Self::Single(Source::Stdin)),
            [path] if !path.is_dir() => Ok(Self::Single(Source::File(path.clone()))),
            _ => {
                ensure!(
                    !paths.iter().any(|path| path == stdin),
                    "`-` (stdin) can't be combined with other paths"
                );
                let mut files = Vec::new();
                for path in paths {
                    if path.is_dir() {
                        files.extend(files_in(path)?);
                    } else {
                        files.push(path.clone());
                    }
                }
                ensure!(!files.is_empty(), "no files to scan");
                Ok(Self::Batch(files))
            }
        }
    }
}

impl Source {
    pub fn read(&self) -> Result<String> {
        match self {
            Self::Stdin => io::read_to_string(io::stdin()).context("failed to read stdin"),
            Self::File(path) => fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display())),
        }
    }
}

impl Text {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        Ok(String::from_utf8(bytes).map_or(Self::NotUtf8, Self::Utf8))
    }
}

/// The non-hidden regular files directly in `dir` (following symlinks),
/// sorted by path.
fn files_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let context = || format!("failed to list {}", dir.display());
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).with_context(context)? {
        let path = entry.with_context(context)?.path();
        let hidden = path
            .file_name()
            .is_some_and(|name| name.as_encoded_bytes().starts_with(b"."));
        if !hidden && path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(paths: &[&str]) -> Vec<PathBuf> {
        paths.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn no_path_or_dash_reads_stdin() {
        assert_eq!(
            Input::from_paths(&[]).unwrap(),
            Input::Single(Source::Stdin)
        );
        assert_eq!(
            Input::from_paths(&paths(&["-"])).unwrap(),
            Input::Single(Source::Stdin)
        );
    }

    #[test]
    fn one_file_is_single_even_if_missing() {
        assert_eq!(
            Input::from_paths(&paths(&["no/such.txt"])).unwrap(),
            Input::Single(Source::File("no/such.txt".into()))
        );
    }

    #[test]
    fn several_files_are_a_batch_in_the_given_order() {
        assert_eq!(
            Input::from_paths(&paths(&["b.txt", "a.txt"])).unwrap(),
            Input::Batch(paths(&["b.txt", "a.txt"]))
        );
    }

    #[test]
    fn a_directory_is_a_batch_of_its_sorted_files() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        assert_eq!(
            Input::from_paths(std::slice::from_ref(&dir)).unwrap(),
            Input::Batch(vec![dir.join("human.txt"), dir.join("llm-styled.md")])
        );
    }

    #[test]
    fn stdin_does_not_mix_with_paths() {
        let err = Input::from_paths(&paths(&["-", "a.txt"])).unwrap_err();
        assert!(err.to_string().contains("can't be combined"), "{err}");
    }
}

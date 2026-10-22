use std::fmt;

use anyhow::{Context, Result};

use crate::probability::Probability;
use crate::text::Segment;

/// Characters of segment text shown in the breakdown.
const PREVIEW_CHARS: usize = 48;

/// Per-segment scores of an input and their word-weighted mean.
///
/// The mean reads as the expected share of the text that was LLM-written.
/// `Display` renders the per-segment breakdown.
#[derive(Debug)]
pub struct Report<'a> {
    input: &'a str,
    segments: Vec<Scored<'a>>,
    overall: Probability,
}

#[derive(Debug)]
struct Scored<'a> {
    segment: Segment<'a>,
    llm: Probability,
}

impl<'a> Report<'a> {
    /// Scores each of `input`'s segments by its [unwrapped](Segment::unwrapped)
    /// text with `score`. The segments must be in input order, as
    /// [`crate::segments`] and [`crate::Detector::split_to_fit`] give them.
    /// Fails if any score fails, or if the segments hold no words to weigh
    /// the scores by.
    pub fn new(
        input: &'a str,
        segments: Vec<Segment<'a>>,
        mut score: impl FnMut(&str) -> Result<Probability>,
    ) -> Result<Self> {
        debug_assert!(
            segments
                .iter()
                .all(|s| input.get(s.range.clone()) == Some(s.text)),
            "segments must come from `input`"
        );
        let segments = segments
            .into_iter()
            .map(|segment| {
                let llm = score(&segment.unwrapped())
                    .with_context(|| format!("failed to score lines {:?}", segment.lines))?;
                Ok(Scored { segment, llm })
            })
            .collect::<Result<Vec<_>>>()?;
        let overall = Probability::weighted_mean(segments.iter().map(|s| (s.llm, s.segment.words)))
            .context("no words to score")?;
        Ok(Self {
            input,
            segments,
            overall,
        })
    }

    /// The text the segments were taken from.
    pub fn input(&self) -> &'a str {
        self.input
    }

    pub fn overall(&self) -> Probability {
        self.overall
    }

    /// Each scored window, in input order.
    pub fn windows(&self) -> impl Iterator<Item = (&Segment<'a>, Probability)> {
        self.segments.iter().map(|s| (&s.segment, s.llm))
    }
}

impl fmt::Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:<9} {:>4} {:>6}  text", "lines", "llm", "words")?;
        for scored in &self.segments {
            let lines = &scored.segment.lines;
            let lines = if lines.start() == lines.end() {
                lines.start().to_string()
            } else {
                format!("{}-{}", lines.start(), lines.end())
            };
            writeln!(
                f,
                "{lines:<9} {:>4} {:>6}  {}",
                scored.llm,
                scored.segment.words,
                preview(scored.segment.text)
            )?;
        }
        let total: usize = self.segments.iter().map(|s| s.segment.words).sum();
        writeln!(f, "{:<9} {:>4} {total:>6}", "overall", self.overall)
    }
}

/// The start of `text` on one line, with whitespace collapsed.
fn preview(text: &str) -> String {
    let mut out = String::with_capacity(PREVIEW_CHARS + 3);
    let mut chars = 0;
    for word in text.split_whitespace() {
        let sep = usize::from(chars > 0);
        let len = word.chars().count();
        if chars + sep + len > PREVIEW_CHARS {
            let room = PREVIEW_CHARS.saturating_sub(chars + sep);
            if sep == 1 && room > 0 {
                out.push(' ');
            }
            out.extend(word.chars().take(room));
            out.push('…');
            break;
        }
        if sep == 1 {
            out.push(' ');
        }
        out.push_str(word);
        chars += sep + len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One segment per blank-line separated paragraph of `input`, counted as
    /// `words[i]` words each.
    fn paragraphs<'a>(input: &'a str, words: &[usize]) -> Vec<Segment<'a>> {
        let mut start = 0;
        input
            .split("\n\n")
            .zip(words)
            .map(|(text, &words)| {
                let line = 1 + input[..start].matches('\n').count();
                let segment = Segment {
                    text,
                    range: start..start + text.len(),
                    lines: line..=line,
                    words,
                };
                start += text.len() + 2;
                segment
            })
            .collect()
    }

    fn certain(llm: bool) -> Probability {
        Probability::from_logit(if llm { 1e6 } else { -1e6 })
    }

    /// Scores text containing "LLM" as certainly LLM-written, all else human.
    fn by_marker(text: &str) -> Result<Probability> {
        Ok(certain(text.contains("LLM")))
    }

    #[test]
    fn overall_is_the_word_weighted_mean() {
        let input = "LLM\n\nhuman";
        let report = Report::new(input, paragraphs(input, &[300, 100]), by_marker).unwrap();
        assert_eq!(report.overall().to_string(), "75%");
    }

    #[test]
    fn breakdown_lists_each_segment_then_the_total() {
        let input = "LLM text\n\nhuman text";
        let report = Report::new(input, paragraphs(input, &[300, 100]), by_marker).unwrap();
        assert_eq!(
            report.to_string(),
            "lines      llm  words  text\n\
             1         100%    300  LLM text\n\
             3           0%    100  human text\n\
             overall    75%    400\n"
        );
    }

    #[test]
    fn scoring_errors_name_the_segment() {
        let input = "a\n\nb";
        let err = Report::new(input, paragraphs(input, &[1, 1]), |text| match text {
            "b" => anyhow::bail!("boom"),
            _ => Ok(certain(false)),
        })
        .unwrap_err();
        assert_eq!(format!("{err:#}"), "failed to score lines 3..=3: boom");
    }

    #[test]
    fn no_words_no_report() {
        assert!(Report::new("", vec![], by_marker).is_err());
        assert!(Report::new("…", paragraphs("…", &[0]), by_marker).is_err());
    }

    #[test]
    fn preview_collapses_whitespace_and_truncates() {
        assert_eq!(preview("  a\n b  c "), "a b c");
        let long = "word ".repeat(20);
        let got = preview(&long);
        assert_eq!(got.chars().count(), PREVIEW_CHARS + 1);
        assert!(got.ends_with('…'));
    }

    #[test]
    fn preview_counts_chars_not_bytes() {
        let got = preview(&"é".repeat(PREVIEW_CHARS + 5));
        assert_eq!(got.chars().count(), PREVIEW_CHARS + 1);
    }
}

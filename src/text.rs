//! Splitting input into windows of prose for the detector.

use std::borrow::Cow;
use std::ops::{Range, RangeInclusive};

use anyhow::Result;

/// Paragraphs are merged until a window has this many words: the detector
/// is unreliable on short snippets.
const MIN_WINDOW_WORDS: usize = 100;

/// Paragraphs longer than this are split at sentence boundaries, keeping
/// windows near the 50–300 word texts the detector was evaluated on.
const MAX_CHUNK_WORDS: usize = 300;

/// Tokens that end in a period without ending a sentence.
const ABBREVIATIONS: &[&str] = &[
    "e.g", "i.e", "etc", "vs", "cf", "mr", "mrs", "ms", "dr", "st",
];

/// A window of prose from the input, scored as a unit.
#[derive(Debug, PartialEq)]
pub struct Segment<'a> {
    pub text: &'a str,
    /// Byte range of `text` in the input.
    pub range: Range<usize>,
    /// 1-based line numbers in the input.
    pub lines: RangeInclusive<usize>,
    pub words: usize,
}

impl<'a> Segment<'a> {
    /// The text with hard line wraps inside paragraphs joined by spaces, as
    /// Markdown renders them. The detector reads hard-wrapped prose, such as
    /// 72-column plain text, as machine-written. Blank lines and lines that
    /// start a list item, heading, quote or table row keep their line break.
    pub fn unwrapped(&self) -> Cow<'a, str> {
        if !self.text.contains('\n') {
            return Cow::Borrowed(self.text);
        }
        let mut out = String::with_capacity(self.text.len());
        let mut prev: Option<&str> = None;
        for line in self.text.lines().map(str::trim) {
            if let Some(prev) = prev {
                let soft = !prev.is_empty()
                    && !line.is_empty()
                    && !starts_block(line)
                    && !is_single_line_block(prev);
                out.push(if soft { ' ' } else { '\n' });
            }
            out.push_str(line);
            prev = Some(line);
        }
        Cow::Owned(out)
    }

    /// Splits the segment into consecutive pieces whose unwrapped text
    /// measures at most `limit` by `size` (tokens, say), so that the detector
    /// can read each whole. Pieces end at sentence boundaries, or at word
    /// boundaries inside a sentence too long by itself, and are as equal in
    /// size as those allow. A single word too long stays as it is.
    pub fn split_to_fit(
        self,
        limit: usize,
        mut size: impl FnMut(&str) -> Result<usize>,
    ) -> Result<Vec<Self>> {
        let mut pieces = Vec::new();
        self.split_into(limit, &mut size, &mut pieces)?;
        Ok(pieces)
    }

    fn split_into(
        self,
        limit: usize,
        size: &mut impl FnMut(&str) -> Result<usize>,
        pieces: &mut Vec<Self>,
    ) -> Result<()> {
        let total = size(&self.unwrapped())?;
        if total <= limit {
            pieces.push(self);
            return Ok(());
        }
        let mut units: Vec<_> = sentence_spans(self.text).collect();
        if units.len() < 2 {
            units = word_spans(self.text);
        }
        if units.len() < 2 {
            pieces.push(self);
            return Ok(());
        }
        let sizes = units
            .iter()
            .map(|unit| size(&self.text[unit.clone()]))
            .collect::<Result<Vec<_>>>()?;
        // Sizes of separate units only approximate the size of their run, so
        // each piece is measured again, and split again if still too long.
        let mut start = 0;
        let runs = total.div_ceil(limit.max(1)).min(units.len());
        for end in balanced_cuts(&sizes, runs) {
            self.piece(units[start].start..units[end - 1].end)
                .split_into(limit, size, pieces)?;
            start = end;
        }
        Ok(())
    }

    /// The part of the segment at `range`, relative to its text, which must
    /// not end in a line break.
    fn piece(&self, range: Range<usize>) -> Self {
        let text = &self.text[range.clone()];
        let first = self.lines.start() + self.text[..range.start].matches('\n').count();
        Segment {
            text,
            range: self.range.start + range.start..self.range.start + range.end,
            lines: first..=first + text.matches('\n').count(),
            words: words(text).count(),
        }
    }
}

/// End indices of `runs` consecutive, non-empty runs of `sizes` whose sums
/// are as close to equal as the boundaries between them allow. `runs` must
/// be between 1 and the number of sizes.
fn balanced_cuts(sizes: &[usize], runs: usize) -> Vec<usize> {
    let prefix: Vec<usize> = std::iter::once(0)
        .chain(sizes.iter().scan(0, |sum, &size| {
            *sum += size;
            Some(*sum)
        }))
        .collect();
    let total = prefix[sizes.len()];
    let mut cuts = Vec::with_capacity(runs);
    let mut previous = 0;
    for run in 1..runs {
        let ideal = total * run / runs;
        // Leaves at least one unit for this run and for each one after it.
        let (lo, hi) = (previous + 1, sizes.len() - (runs - run));
        let past = prefix.partition_point(|&sum| sum < ideal).clamp(lo, hi);
        let cut = if past > lo && ideal.abs_diff(prefix[past - 1]) < ideal.abs_diff(prefix[past]) {
            past - 1
        } else {
            past
        };
        cuts.push(cut);
        previous = cut;
    }
    cuts.push(sizes.len());
    cuts
}

/// Byte ranges of the runs of non-whitespace in `text`.
fn word_spans(text: &str) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (true, Some(s)) => {
                spans.push(s..i);
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    spans.extend(start.map(|s| s..text.len()));
    spans
}

/// Whether a trimmed line opens a Markdown block: a list item, heading,
/// quote, table row or thematic break.
fn starts_block(line: &str) -> bool {
    let after_digits = line.trim_start_matches(|c: char| c.is_ascii_digit());
    let ordinal = after_digits.len() < line.len()
        && (after_digits.starts_with(". ") || after_digits.starts_with(") "));
    ordinal
        || ["- ", "* ", "+ ", "• ", ">"]
            .iter()
            .any(|marker| line.starts_with(marker))
        || is_single_line_block(line)
}

/// Whether a trimmed line is a block that never continues on the next line:
/// a heading, table row or thematic break (`---`, `***`, `___`).
fn is_single_line_block(line: &str) -> bool {
    let thematic_break = line.len() >= 3
        && ['-', '*', '_']
            .iter()
            .any(|&mark| line.chars().all(|c| c == mark || c == ' '));
    line.starts_with('#') || line.starts_with('|') || thematic_break
}

/// Splits `text` into windows of about 100–400 words, skipping fenced code
/// blocks. Windows never span a code block; a short tail is folded into the
/// window before it.
pub fn segments(text: &str) -> Vec<Segment<'_>> {
    let mut windows = WindowBuilder::default();
    let mut paragraph: Option<Range<usize>> = None;
    let mut in_code = false;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let start = offset;
        offset += line.len();
        let trimmed = line.trim();
        let is_fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");
        if is_fence || in_code || trimmed.is_empty() {
            if let Some(range) = paragraph.take() {
                windows.push_paragraph(text, range);
            }
            if is_fence {
                in_code = !in_code;
                windows.barrier();
            }
            continue;
        }
        let end = start + line.trim_end().len();
        match &mut paragraph {
            Some(range) => range.end = end,
            None => paragraph = Some(start..end),
        }
    }
    if let Some(range) = paragraph {
        windows.push_paragraph(text, range);
    }
    let lines = LineIndex::new(text);
    windows
        .finish()
        .into_iter()
        .map(|span| Segment {
            text: &text[span.range.clone()],
            lines: lines.of(&span.range),
            range: span.range,
            words: span.words,
        })
        .collect()
}

/// Splits `text` into words, stripping surrounding punctuation and dropping
/// tokens without letters (numbers, bullets, emoji).
fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| c.is_whitespace() || matches!(c, '—' | '–'))
        .map(|token| token.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|word| word.chars().any(char::is_alphabetic))
}

/// Byte ranges of the trimmed sentences in `text`. Line breaks always end a
/// sentence, so list items and headings count as sentences of their own.
fn sentence_spans(text: &str) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        while start < text.len() {
            let end = start + sentence_end(&text[start..]);
            let sentence = &text[start..end];
            let lead = sentence.len() - sentence.trim_start().len();
            let len = sentence.trim().len();
            let range = start + lead..start + lead + len;
            start = end;
            if len > 0 {
                return Some(range);
            }
        }
        None
    })
}

/// Byte offset just past the first sentence of `text`, or its length when no
/// sentence ends in it.
fn sentence_end(text: &str) -> usize {
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '\n' => return i + 1,
            '.' | '!' | '?' => {
                // Swallow repeated terminators and closers: `?!`, `..."`, `.)`.
                let mut end = i + c.len_utf8();
                while let Some(&(j, next)) = chars.peek() {
                    if !matches!(next, '.' | '!' | '?' | '"' | '\'' | '”' | '’' | ')' | ']') {
                        break;
                    }
                    end = j + next.len_utf8();
                    chars.next();
                }
                let at_break = chars.peek().is_none_or(|&(_, next)| next.is_whitespace());
                if at_break && !ends_with_abbreviation(&text[..i]) {
                    return end;
                }
            }
            _ => {}
        }
    }
    text.len()
}

fn ends_with_abbreviation(text: &str) -> bool {
    let last = text
        .rsplit(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or_default();
    ABBREVIATIONS.iter().any(|a| a.eq_ignore_ascii_case(last))
}

/// A byte range of the input and its word count.
#[derive(Debug)]
struct Span {
    range: Range<usize>,
    words: usize,
}

impl Span {
    fn new(text: &str, range: Range<usize>) -> Self {
        let words = words(&text[range.clone()]).count();
        Self { range, words }
    }

    /// Grows `self` to also cover `later`, which must follow it in the input.
    fn extend(&mut self, later: Span) {
        self.range.end = later.range.end;
        self.words += later.words;
    }
}

/// Splits the paragraph at `range` into chunks of at most [`MAX_CHUNK_WORDS`]
/// words at sentence boundaries. A single longer sentence stays whole.
fn chunks(text: &str, range: Range<usize>) -> Vec<Span> {
    let whole = Span::new(text, range.clone());
    if whole.words <= MAX_CHUNK_WORDS {
        return vec![whole];
    }
    let mut out = Vec::new();
    let mut current: Option<Span> = None;
    for sentence in sentence_spans(&text[range.clone()]) {
        let sentence = Span::new(
            text,
            range.start + sentence.start..range.start + sentence.end,
        );
        match &mut current {
            Some(chunk) if chunk.words + sentence.words <= MAX_CHUNK_WORDS => {
                chunk.extend(sentence)
            }
            _ => out.extend(current.replace(sentence)),
        }
    }
    out.extend(current);
    out
}

/// Accumulates chunks into windows of at least [`MIN_WINDOW_WORDS`] words.
#[derive(Debug, Default)]
struct WindowBuilder {
    done: Vec<Span>,
    pending: Option<Span>,
    /// Whether the last finished window is contiguous with `pending`, so a
    /// short tail can be folded into it.
    can_extend_last: bool,
}

impl WindowBuilder {
    fn push_paragraph(&mut self, text: &str, range: Range<usize>) {
        for chunk in chunks(text, range) {
            self.push(chunk);
        }
    }

    fn push(&mut self, chunk: Span) {
        if chunk.words == 0 {
            return;
        }
        let pending = match self.pending.take() {
            Some(mut pending) => {
                pending.extend(chunk);
                pending
            }
            None => chunk,
        };
        if pending.words >= MIN_WINDOW_WORDS {
            self.done.push(pending);
            self.can_extend_last = true;
        } else {
            self.pending = Some(pending);
        }
    }

    /// Ends the current window for good, e.g. at a code fence.
    fn barrier(&mut self) {
        if let Some(pending) = self.pending.take() {
            match self.done.last_mut() {
                Some(last) if self.can_extend_last => last.extend(pending),
                _ => self.done.push(pending),
            }
        }
        self.can_extend_last = false;
    }

    fn finish(mut self) -> Vec<Span> {
        self.barrier();
        self.done
    }
}

/// Maps byte offsets to 1-based line numbers.
struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    fn new(text: &str) -> Self {
        let starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        Self { starts }
    }

    /// Lines covered by the non-empty `range`.
    fn of(&self, range: &Range<usize>) -> RangeInclusive<usize> {
        let line = |offset: usize| self.starts.partition_point(|&start| start <= offset);
        line(range.start)..=line(range.end - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence_list(text: &str) -> Vec<&str> {
        sentence_spans(text).map(|r| &text[r]).collect()
    }

    /// `n` words of filler ending in a full stop.
    fn prose(n: usize) -> String {
        format!("{}word.", "word ".repeat(n - 1))
    }

    #[test]
    fn sentences_split_on_terminators_and_newlines() {
        assert_eq!(
            sentence_list("One two. Three?! Four\n- five\nsix..."),
            ["One two.", "Three?!", "Four", "- five", "six..."]
        );
    }

    #[test]
    fn sentences_keep_abbreviations_decimals_and_dotted_names() {
        assert_eq!(
            sentence_list("Use e.g. Node.js vs. Deno at 3.5 GHz. Done"),
            ["Use e.g. Node.js vs. Deno at 3.5 GHz.", "Done"]
        );
    }

    #[test]
    fn sentences_include_closing_quotes() {
        assert_eq!(
            sentence_list(r#"He said "stop." then left."#),
            [r#"He said "stop.""#, "then left."]
        );
    }

    #[test]
    fn sentences_of_blank_text_are_empty() {
        assert!(sentence_list(" \n\n ").is_empty());
    }

    #[test]
    fn words_strip_punctuation_and_drop_non_words() {
        let got: Vec<_> = words("**Speed**: it's 42 — fast-paced (really)! - ✅").collect();
        assert_eq!(got, ["Speed", "it's", "fast-paced", "really"]);
    }

    #[test]
    fn words_split_on_unspaced_dashes() {
        let got: Vec<_> = words("growth—and change–fast").collect();
        assert_eq!(got, ["growth", "and", "change", "fast"]);
    }

    #[test]
    fn short_paragraphs_merge_until_the_minimum() {
        let text = format!("# Title\n\n{}\n\n{}\n", prose(60), prose(60));
        let got = segments(&text);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].lines, 1..=5);
        assert_eq!(got[0].words, 121);
    }

    #[test]
    fn windows_close_once_they_reach_the_minimum() {
        let body = prose(MIN_WINDOW_WORDS);
        let text = format!("{body}\n\n{body}\n");
        let got = segments(&text);
        assert_eq!(
            got.iter().map(|s| s.lines.clone()).collect::<Vec<_>>(),
            [1..=1, 3..=3]
        );
        assert!(got.iter().all(|s| s.words == MIN_WINDOW_WORDS));
    }

    #[test]
    fn short_tail_folds_into_previous_window() {
        let text = format!("{}\n\nThe end.\n", prose(MIN_WINDOW_WORDS));
        let got = segments(&text);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].lines, 1..=3);
        assert!(got[0].text.ends_with("The end."));
    }

    #[test]
    fn long_paragraphs_split_at_sentences() {
        let sentence = prose(50);
        let paragraph = [sentence.as_str(); 8].join(" "); // 400 words
        let got = segments(&paragraph);
        assert_eq!(got.iter().map(|s| s.words).collect::<Vec<_>>(), [300, 100]);
        assert!(got[0].text.ends_with("word."));
        assert!(got[1].text.starts_with("word"));
        assert_eq!(got[0].text.len() + 1 + got[1].text.len(), paragraph.len());
    }

    #[test]
    fn a_single_overlong_sentence_stays_whole() {
        let sentence = "word ".repeat(MAX_CHUNK_WORDS + 50);
        let got = segments(&sentence);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].words, MAX_CHUNK_WORDS + 50);
    }

    #[test]
    fn code_blocks_are_skipped_and_never_merged_across() {
        let text = "Before:\n```\nlet x = 1;\n```\nAfter.\n";
        let got = segments(text);
        assert_eq!(
            got,
            [
                Segment {
                    text: "Before:",
                    range: 0..7,
                    lines: 1..=1,
                    words: 1
                },
                Segment {
                    text: "After.",
                    range: 27..33,
                    lines: 5..=5,
                    words: 1
                },
            ]
        );
    }

    #[test]
    fn unterminated_code_block_runs_to_the_end() {
        assert!(segments("```\nnot prose\n").is_empty());
    }

    #[test]
    fn handles_crlf_line_endings() {
        let got = segments("one\r\n\r\ntwo\r\n");
        assert_eq!(
            got,
            [Segment {
                text: "one\r\n\r\ntwo",
                range: 0..10,
                lines: 1..=3,
                words: 2
            }]
        );
    }

    #[test]
    fn text_without_words_has_no_segments() {
        assert!(segments("").is_empty());
        assert!(segments("\n\n").is_empty());
        assert!(segments("42 — 7\n\n✅\n").is_empty());
    }

    fn unwrapped(text: &str) -> Cow<'_, str> {
        Segment {
            text,
            range: 0..text.len(),
            lines: 1..=1,
            words: 0,
        }
        .unwrapped()
    }

    #[test]
    fn unwrapping_joins_soft_breaks_and_keeps_paragraphs() {
        assert_eq!(
            unwrapped("one\n  two\nthree\n\nfour\r\nfive"),
            "one two three\n\nfour five"
        );
    }

    #[test]
    fn unwrapping_keeps_block_structure() {
        let text = "Intro:\n- one\n- two\n  continued\n1. first\n2) second\n# Head\nbody\n> quote\n| a | b |\n---\nend";
        assert_eq!(
            unwrapped(text),
            "Intro:\n- one\n- two continued\n1. first\n2) second\n# Head\nbody\n> quote\n| a | b |\n---\nend"
        );
    }

    #[test]
    fn unwrapping_a_single_line_borrows() {
        assert!(matches!(unwrapped("no breaks here"), Cow::Borrowed(_)));
    }

    #[test]
    fn numbers_and_dashes_inside_prose_do_not_start_blocks() {
        assert_eq!(unwrapped("in\n1883 the\n-- river"), "in 1883 the -- river");
    }

    /// `text` as a segment of an input, at byte 10 and line 3.
    fn segment(text: &str) -> Segment<'_> {
        Segment {
            text,
            range: 10..10 + text.len(),
            lines: 3..=3 + text.matches('\n').count(),
            words: words(text).count(),
        }
    }

    /// Splits `text` into pieces of at most `limit` words.
    fn split(text: &str, limit: usize) -> Vec<Segment<'_>> {
        segment(text)
            .split_to_fit(limit, |text| Ok(words(text).count()))
            .unwrap()
    }

    #[test]
    fn balanced_cuts_split_sums_evenly_at_boundaries() {
        assert_eq!(balanced_cuts(&[1; 10], 2), [5, 10]);
        assert_eq!(balanced_cuts(&[10; 10], 3), [3, 7, 10]);
        assert_eq!(balanced_cuts(&[8, 1, 1], 2), [1, 3]);
        assert_eq!(balanced_cuts(&[1, 1, 8], 2), [2, 3]);
        assert_eq!(balanced_cuts(&[5, 5], 2), [1, 2]);
        assert_eq!(balanced_cuts(&[3, 4], 1), [2]);
    }

    #[test]
    fn a_segment_that_fits_stays_whole() {
        let text = prose(40);
        assert_eq!(split(&text, 40), [segment(&text)]);
    }

    #[test]
    fn a_long_segment_splits_evenly_at_sentences() {
        let sentence = prose(10);
        let text = [sentence.as_str(); 10].join(" "); // 100 words
        let got = split(&text, 40);
        assert_eq!(
            got.iter().map(|s| s.words).collect::<Vec<_>>(),
            [30, 40, 30]
        );
        assert!(
            got.iter()
                .all(|s| s.text.starts_with("word") && s.text.ends_with('.'))
        );
        // Each piece is the input at its range; only the spaces between go.
        for piece in &got {
            assert_eq!(
                &text[piece.range.start - 10..piece.range.end - 10],
                piece.text
            );
        }
        let joined: Vec<_> = got.iter().map(|s| s.text).collect();
        assert_eq!(joined.join(" "), text);
    }

    #[test]
    fn pieces_keep_their_line_numbers() {
        let line = prose(10);
        let text = [line.as_str(); 6].join("\n"); // input lines 3–8
        let got = split(&text, 30);
        let lines: Vec<_> = got.iter().map(|s| s.lines.clone()).collect();
        assert_eq!(lines, [3..=5, 6..=8]);
    }

    #[test]
    fn a_long_sentence_splits_at_words() {
        let text = "word ".repeat(99) + "word";
        let got = split(&text, 40);
        assert_eq!(
            got.iter().map(|s| s.words).collect::<Vec<_>>(),
            [33, 33, 34]
        );
    }

    #[test]
    fn mixed_pieces_split_again_until_they_fit() {
        // A long sentence between short ones: split at sentences, then words.
        let text = format!("{} {} {}", prose(5), prose(60), prose(5));
        let got = split(&text, 30);
        assert_eq!(
            got.iter().map(|s| s.words).collect::<Vec<_>>(),
            [5, 30, 30, 5]
        );
    }

    #[test]
    fn a_single_word_too_long_stays_whole() {
        let text = "supercalifragilistic";
        let got = segment(text)
            .split_to_fit(5, |text| Ok(text.len()))
            .unwrap();
        assert_eq!(got, [segment(text)]);
    }

    #[test]
    fn line_index_maps_offsets() {
        let index = LineIndex::new("ab\ncd\n\nef");
        assert_eq!(index.of(&(0..2)), 1..=1);
        assert_eq!(index.of(&(3..9)), 2..=4);
        assert_eq!(index.of(&(2..3)), 1..=1); // the newline belongs to its line
    }
}

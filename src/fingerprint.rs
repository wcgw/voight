//! Traces of how a text was made that the detector can't weigh: characters
//! that disguise text from detectors, which [`normalize`] undoes before the
//! detector reads it, and residue that chatbot interfaces leave in text
//! copied from them.

use std::borrow::Cow;
use std::iter;

use unicode_normalization::UnicodeNormalization;

use crate::text::Segment;

/// Text that chatbot interfaces leave in answers copied from them. Matched
/// ignoring ASCII case, with each space matching any run of whitespace.
const RESIDUE: &[&str] = &[
    // ChatGPT's references to the web pages an answer draws on, meant for
    // its interface, and the tracking parameter it adds to the links it
    // cites.
    "contentReference[oaicite:",
    "turn0search",
    "utm_source=chatgpt.com",
    // Chatbot disclaimers, found pasted into published papers.
    "As an AI language model",
    "As of my last knowledge update",
];

/// `text` as a detector should read it, with the ways of disguising text
/// from detectors undone: invisible characters are removed, unusual spaces
/// (no-break, thin, ideographic, …) become plain ones, compatibility forms
/// of letters (fullwidth, mathematical, ligatures) become plain letters, and
/// letters of other scripts that look like Latin ones (Cyrillic `а` for `a`)
/// become Latin inside words that hold Latin letters too. Text with none of
/// these is returned as it is.
///
/// Emoji keep the invisible characters that join them into one (`👩‍💻`) or
/// select their style (`⚠️`), and words wholly in another script keep their
/// letters.
pub fn normalize(text: &str) -> Cow<'_, str> {
    undisguise(text).0
}

/// Traces of how a text was made, found in its windows: what [`normalize`]
/// undid that points to a disguise, and chatbot residue.
#[derive(Debug, Default, PartialEq)]
pub struct Fingerprints {
    undone: Undone,
    /// In order of the lines they first occur on.
    residue: Vec<Residue>,
}

/// A [`RESIDUE`] marker found in a text.
#[derive(Debug, PartialEq)]
struct Residue {
    marker: &'static str,
    /// 1-based line of its first occurrence.
    line: usize,
    occurrences: usize,
}

impl Fingerprints {
    /// The fingerprints in `segments`, which must be in input order.
    pub fn find<'s, 'a: 's>(segments: impl IntoIterator<Item = &'s Segment<'a>>) -> Self {
        let mut found = Self::default();
        for segment in segments {
            let (_, undone) = undisguise(segment.text);
            found.undone.invisible += undone.invisible;
            found.undone.lookalikes += undone.lookalikes;
            for &marker in RESIDUE {
                for at in occurrences(segment.text, marker) {
                    match found.residue.iter_mut().find(|r| r.marker == marker) {
                        Some(residue) => residue.occurrences += 1,
                        None => found.residue.push(Residue {
                            marker,
                            line: segment.lines.start() + segment.text[..at].matches('\n').count(),
                            occurrences: 1,
                        }),
                    }
                }
            }
        }
        found.residue.sort_by_key(|residue| residue.line);
        found
    }

    /// One line of text per kind of fingerprint found; nothing when none was.
    pub fn notes(&self) -> impl Iterator<Item = String> + '_ {
        let Undone {
            invisible,
            lookalikes,
        } = self.undone;
        let undone = [
            (invisible > 0).then(|| {
                format!(
                    "{} inside or between words, removed before scoring",
                    counted(invisible, "invisible character")
                )
            }),
            (lookalikes > 0).then(|| {
                format!(
                    "{} from other scripts in Latin words, replaced before scoring",
                    counted(lookalikes, "look-alike letter")
                )
            }),
        ];
        let residue = self.residue.iter().map(|residue| {
            let mut note = format!(
                "chatbot residue on line {}: {}",
                residue.line, residue.marker
            );
            if residue.occurrences > 1 {
                note += &format!(" ({} times)", residue.occurrences);
            }
            note
        });
        undone.into_iter().flatten().chain(residue)
    }
}

/// `n` and `noun`, plural unless `n` is 1.
fn counted(n: usize, noun: &str) -> String {
    format!("{n} {noun}{}", if n == 1 { "" } else { "s" })
}

/// What [`undisguise`] changed in a text that points to a disguise; stray
/// ones are left by copy and paste, or by text decoded with the wrong
/// encoding.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Undone {
    /// Invisible characters removed after a Latin letter or whitespace.
    /// Emoji hold invisible characters too, after a symbol, as do other
    /// scripts after their letters, and a byte order mark starts a text.
    invisible: usize,
    /// Letters of other scripts replaced by the Latin letters they look
    /// like.
    lookalikes: usize,
}

/// [`normalize`]'s result, and what it changed that points to a
/// disguise.
fn undisguise(text: &str) -> (Cow<'_, str>, Undone) {
    if text.chars().all(is_plain) {
        return (Cow::Borrowed(text), Undone::default());
    }
    let mut undone = Undone::default();
    let mut plain = String::with_capacity(text.len());
    let mut last_visible: Option<char> = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if is_invisible(c) {
            match last_visible {
                Some(last) if is_latin(last) || last.is_whitespace() => {
                    undone.invisible += 1;
                }
                Some(_) if joins_emoji(c) => plain.push(c),
                _ => {}
            }
            continue;
        }
        last_visible = Some(c);
        if is_unusual_space(c) {
            plain.push(' ');
        } else if c.is_alphabetic() && !c.is_ascii() && chars.peek() != Some(&EMOJI_STYLE) {
            plain.extend(iter::once(c).nfkc());
        } else {
            plain.push(c);
        }
    }
    let plain = replace_lookalikes(&plain, &mut undone);
    (Cow::Owned(plain), undone)
}

/// `text` with the letters of other scripts that look like Latin ones
/// replaced by them, in words that hold Latin letters too. Counts the
/// replacements in `undone`.
fn replace_lookalikes(text: &str, undone: &mut Undone) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let gap = rest.find(char::is_alphabetic).unwrap_or(rest.len());
        out.push_str(&rest[..gap]);
        rest = &rest[gap..];
        let len = rest
            .find(|c: char| !c.is_alphabetic())
            .unwrap_or(rest.len());
        let (word, after) = rest.split_at(len);
        if word.chars().any(is_latin) && word.chars().any(|c| latin_twin(c).is_some()) {
            for c in word.chars() {
                match latin_twin(c) {
                    Some(twin) => {
                        out.push(twin);
                        undone.lookalikes += 1;
                    }
                    None => out.push(c),
                }
            }
        } else {
            out.push_str(word);
        }
        rest = after;
    }
    out
}

/// Whether [`normalize`] leaves `c` as it is wherever it occurs.
fn is_plain(c: char) -> bool {
    c.is_ascii()
        || !(is_invisible(c)
            || is_unusual_space(c)
            || latin_twin(c).is_some()
            || (c.is_alphabetic() && !iter::once(c).nfkc().eq(iter::once(c))))
}

/// Whether `c` is a default ignorable code point: a character that is
/// invisible unless it combines with others, per
/// [DerivedCoreProperties.txt](https://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt)
/// in Unicode 18.0.
fn is_invisible(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}'
            | '\u{034F}'
            | '\u{061C}'
            | '\u{115F}'..='\u{1160}'
            | '\u{17B4}'..='\u{17B5}'
            | '\u{180B}'..='\u{180F}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{206F}'
            | '\u{3164}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FEFF}'
            | '\u{FFA0}'
            | '\u{FFF0}'..='\u{FFF8}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0000}'..='\u{E0FFF}'
    )
}

/// The variation selector that asks for the character before it to be shown
/// as an emoji, as letter-like ones can be (`ℹ️`, `Ⓜ️`).
const EMOJI_STYLE: char = '\u{FE0F}';

/// Whether `c`, an invisible character, can join emoji into one or select
/// their style: the zero width joiner, variation selectors and tags.
fn joins_emoji(c: char) -> bool {
    matches!(
        c,
        '\u{200D}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{E0001}'
            | '\u{E0020}'..='\u{E007F}'
            | '\u{E0100}'..='\u{E01EF}'
    )
}

/// Whether `c` is a space separator other than the plain space.
fn is_unusual_space(c: char) -> bool {
    matches!(
        c,
        '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// Whether `c` is a Latin letter (in the Basic Latin, Latin-1 and Latin
/// Extended blocks).
fn is_latin(c: char) -> bool {
    c.is_ascii_alphabetic()
        || (c.is_alphabetic() && matches!(c, '\u{00C0}'..='\u{024F}' | '\u{1E00}'..='\u{1EFF}'))
}

/// The Latin letter that `c`, a Cyrillic, Greek or Armenian letter, looks
/// like: the ones that tools hiding text from detectors substitute.
fn latin_twin(c: char) -> Option<char> {
    Some(match c {
        // Greek
        '\u{0391}' => 'A',
        '\u{0392}' => 'B',
        '\u{0395}' => 'E',
        '\u{0396}' => 'Z',
        '\u{0397}' => 'H',
        '\u{0399}' => 'I',
        '\u{039A}' => 'K',
        '\u{039C}' => 'M',
        '\u{039D}' => 'N',
        '\u{039F}' => 'O',
        '\u{03A1}' => 'P',
        '\u{03A4}' => 'T',
        '\u{03A5}' => 'Y',
        '\u{03A7}' => 'X',
        '\u{03B1}' => 'a',
        '\u{03B9}' => 'i',
        '\u{03BD}' => 'v',
        '\u{03BF}' => 'o',
        '\u{03C1}' => 'p',
        '\u{03C5}' => 'u',
        // Cyrillic
        '\u{0405}' => 'S',
        '\u{0406}' => 'I',
        '\u{0408}' => 'J',
        '\u{0410}' => 'A',
        '\u{0412}' => 'B',
        '\u{0415}' => 'E',
        '\u{041A}' => 'K',
        '\u{041C}' => 'M',
        '\u{041D}' => 'H',
        '\u{041E}' => 'O',
        '\u{0420}' => 'P',
        '\u{0421}' => 'C',
        '\u{0422}' => 'T',
        '\u{0425}' => 'X',
        '\u{0430}' => 'a',
        '\u{0435}' => 'e',
        '\u{043E}' => 'o',
        '\u{0440}' => 'p',
        '\u{0441}' => 'c',
        '\u{0443}' => 'y',
        '\u{0445}' => 'x',
        '\u{0455}' => 's',
        '\u{0456}' => 'i',
        '\u{0458}' => 'j',
        '\u{04AE}' => 'Y',
        '\u{04BB}' => 'h',
        '\u{04C0}' => 'I',
        '\u{04CF}' => 'l',
        '\u{0501}' => 'd',
        '\u{051A}' => 'Q',
        '\u{051B}' => 'q',
        '\u{051C}' => 'W',
        '\u{051D}' => 'w',
        // Armenian
        '\u{0570}' => 'h',
        '\u{0578}' => 'n',
        '\u{057D}' => 'u',
        '\u{0585}' => 'o',
        _ => return None,
    })
}

/// Byte offsets in `text` where `marker` starts, ignoring ASCII case and
/// matching each space in `marker` with any run of whitespace, so that a
/// marker wrapped over lines is found.
fn occurrences<'t>(text: &'t str, marker: &'t str) -> impl Iterator<Item = usize> + 't {
    text.char_indices()
        .map(|(at, _)| at)
        .filter(move |&at| starts_with_marker(&text[at..], marker))
}

fn starts_with_marker(text: &str, marker: &str) -> bool {
    let mut rest = text;
    for (i, part) in marker.split(' ').enumerate() {
        if i > 0 {
            let spaced = rest.trim_start();
            if spaced.len() == rest.len() {
                return false;
            }
            rest = spaced;
        }
        match rest.get(..part.len()) {
            Some(head) if head.eq_ignore_ascii_case(part) => rest = &rest[part.len()..],
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text;

    fn undone(invisible: usize, lookalikes: usize) -> Undone {
        Undone {
            invisible,
            lookalikes,
        }
    }

    #[test]
    fn ordinary_text_is_read_as_it_is() {
        let text = "“Quotes” — dashes… naïve café, 5 m², ½ cup, α = 0.05, \
                    ⚠️ warning, 👩‍💻 at work, 1️⃣ first; Привет, мир; Ελληνικά.";
        assert_eq!(undisguise(text), (Cow::Borrowed(text), Undone::default()));
        assert!(matches!(normalize("plain ASCII"), Cow::Borrowed(_)));
    }

    #[test]
    fn invisible_characters_are_removed_and_counted() {
        // The last one, after punctuation, isn't counted.
        let text = "Ca\u{200B}ching is\u{00AD} hard\u{2060}, mostly.\u{FEFF}";
        assert_eq!(
            undisguise(text),
            (Cow::Owned("Caching is hard, mostly.".into()), undone(3, 0))
        );
    }

    #[test]
    fn invisible_characters_after_a_space_count() {
        assert_eq!(undisguise("a \u{200B}b").1, undone(1, 0));
    }

    #[test]
    fn a_byte_order_mark_is_removed_without_counting() {
        assert_eq!(
            undisguise("\u{FEFF}Text"),
            (Cow::Owned("Text".into()), Undone::default())
        );
    }

    #[test]
    fn joiners_after_a_letter_are_removed() {
        assert_eq!(
            undisguise("wo\u{200D}rd\u{FE0F}"),
            (Cow::Owned("word".into()), undone(2, 0))
        );
    }

    #[test]
    fn joiners_in_other_scripts_are_not_counted() {
        // Persian, with a zero width non-joiner; Hindi, with a joiner.
        let (text, undone) = undisguise("حرف\u{200C}باز क्\u{200D}ष");
        assert_eq!(text, "حرفباز क्\u{200D}ष");
        assert_eq!(undone, Undone::default());
    }

    #[test]
    fn letter_like_emoji_stay_emoji() {
        let text = "\u{2139}\u{FE0F} Note: \u{24C2}\u{FE0F}etro.";
        assert_eq!(undisguise(text), (Cow::Borrowed(text), Undone::default()));
    }

    #[test]
    fn unusual_spaces_become_plain() {
        assert_eq!(
            undisguise("one\u{A0}two\u{2009}three\u{3000}four\u{202F}five"),
            (
                Cow::Owned("one two three four five".into()),
                Undone::default()
            )
        );
    }

    #[test]
    fn styled_letters_become_plain() {
        assert_eq!(
            undisguise("Ｆｕｌｌ 𝐛𝐨𝐥𝐝 ﬁne"),
            (Cow::Owned("Full bold fine".into()), Undone::default())
        );
    }

    #[test]
    fn lookalikes_in_latin_words_become_latin() {
        // Cyrillic е and с, Greek Ο, Armenian օ.
        let text = "Th\u{0435} qui\u{0441}k \u{039F}ld f\u{0585}x";
        assert_eq!(
            undisguise(text),
            (Cow::Owned("The quick Old fox".into()), undone(0, 4))
        );
    }

    #[test]
    fn words_wholly_in_another_script_keep_their_letters() {
        // Russian words of look-alike letters only: сор, а, хор.
        let text = "\u{0441}\u{043E}\u{0440} \u{0430} \u{0445}\u{043E}\u{0440}, and α";
        assert_eq!(
            undisguise(text),
            (Cow::Owned(text.into()), Undone::default())
        );
    }

    #[test]
    fn latin_twins_are_ascii_letters_of_other_scripts() {
        for c in (0..=0x10FFFF).filter_map(char::from_u32) {
            if let Some(twin) = latin_twin(c) {
                assert!(c.is_alphabetic() && !is_latin(c), "{c:?}");
                assert!(twin.is_ascii_alphabetic(), "{c:?}");
                assert_eq!(c.is_uppercase(), twin.is_uppercase(), "{c:?}");
            }
        }
    }

    #[test]
    fn invisible_characters_are_neither_spaces_nor_twins() {
        for c in (0..=0x10FFFF).filter_map(char::from_u32) {
            if is_invisible(c) {
                assert!(!c.is_whitespace() && !is_unusual_space(c), "{c:?}");
                assert!(latin_twin(c).is_none(), "{c:?}");
            }
            if joins_emoji(c) {
                assert!(is_invisible(c), "{c:?}");
            }
        }
    }

    fn fingerprints(input: &str) -> Fingerprints {
        Fingerprints::find(&text::segments(input))
    }

    fn notes(input: &str) -> Vec<String> {
        fingerprints(input).notes().collect()
    }

    #[test]
    fn plain_text_has_no_fingerprints() {
        assert_eq!(
            fingerprints("Some prose.\n\nMore of it."),
            Fingerprints::default()
        );
        assert!(notes("Some prose.").is_empty());
    }

    #[test]
    fn disguises_are_counted_across_windows() {
        let word = "w\u{200B}o\u{0440}d ";
        let input = format!("{}\n\n{}", word.repeat(100), word.repeat(100));
        assert_eq!(text::segments(&input).len(), 2);
        assert_eq!(
            notes(&input),
            [
                "200 invisible characters inside or between words, removed before scoring",
                "200 look-alike letters from other scripts in Latin words, replaced before scoring",
            ]
        );
    }

    #[test]
    fn single_disguises_are_singular() {
        assert_eq!(
            notes("O\u{200B}ne l\u{043E}ok."),
            [
                "1 invisible character inside or between words, removed before scoring",
                "1 look-alike letter from other scripts in Latin words, replaced before scoring",
            ]
        );
    }

    #[test]
    fn residue_is_found_with_its_first_line() {
        let input = "Intro.\n\nSee [docs](https://example.com/?utm_source=chatgpt.com).\n\
                     Also [this](https://example.org/?utm_source=chatgpt.com) \
                     citeturn0search3.\n";
        assert_eq!(
            notes(input),
            [
                "chatbot residue on line 3: utm_source=chatgpt.com (2 times)",
                "chatbot residue on line 4: turn0search",
            ]
        );
    }

    #[test]
    fn residue_matches_any_case_and_wrapped_lines() {
        let input = "Well, as an AI\n  language model, I can't.\n\n\
                     ```\nAs of my last knowledge update\n```\n";
        assert_eq!(
            notes(input),
            ["chatbot residue on line 1: As an AI language model"]
        );
    }

    #[test]
    fn markers_need_whitespace_where_they_have_spaces() {
        assert!(starts_with_marker(
            "as an\tAI language model",
            "As an AI language model"
        ));
        assert!(!starts_with_marker(
            "as anAI language model",
            "As an AI language model"
        ));
        assert!(!starts_with_marker("as an", "As an AI language model"));
        assert_eq!(
            occurrences("é utm_source=chatgpt.com", "utm_source=chatgpt.com").collect::<Vec<_>>(),
            [3]
        );
    }
}

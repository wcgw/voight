//! The input rendered with each window shaded by its score, from green
//! (human) through yellow to red (LLM), in 24-bit terminal colours.

use std::fmt::{self, Write};

use crate::report::Report;

/// Background colours at 0%, 50% and 100%. They're pastel so that the black
/// text drawn on them reads on light and dark terminal themes alike.
const HUMAN: Rgb = Rgb(164, 222, 164);
const EVEN: Rgb = Rgb(246, 229, 141);
const LLM: Rgb = Rgb(244, 150, 150);
const RESET: &str = "\x1b[0m";
/// Scores shown in the footer's colour scale.
const SCALE: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

#[derive(Debug, Clone, Copy, PartialEq)]
struct Rgb(u8, u8, u8);

impl Rgb {
    /// The colour for `score` in `[0, 1]`.
    fn shade(score: f64) -> Self {
        let score = score.clamp(0.0, 1.0);
        let (from, to, t) = if score < 0.5 {
            (HUMAN, EVEN, score * 2.0)
        } else {
            (EVEN, LLM, score * 2.0 - 1.0)
        };
        // Stays within the two endpoints, so the cast can't overflow.
        let mix = |a: u8, b: u8| (f64::from(a) + (f64::from(b) - f64::from(a)) * t).round() as u8;
        Self(mix(from.0, to.0), mix(from.1, to.1), mix(from.2, to.2))
    }
}

/// A report's input with its windows shaded; ends with a footer giving the
/// overall score and the colour scale.
pub struct Heatmap<'r, 'a>(&'r Report<'a>);

impl<'r, 'a> Heatmap<'r, 'a> {
    pub fn new(report: &'r Report<'a>) -> Self {
        Self(report)
    }
}

impl fmt::Display for Heatmap<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let input = self.0.input();
        let mut at = 0;
        for (segment, llm) in self.0.windows() {
            f.write_str(&input[at..segment.range.start])?;
            shaded(f, segment.text, Rgb::shade(llm.value()))?;
            at = segment.range.end;
        }
        f.write_str(&input[at..])?;
        if !input.ends_with('\n') {
            f.write_char('\n')?;
        }
        write!(f, "\n{} overall · 0% ", self.0.overall())?;
        for score in SCALE {
            shaded(f, "  ", Rgb::shade(score))?;
        }
        writeln!(f, " 100%")
    }
}

/// Writes `text` in black on `colour`, closing the colour before each line
/// break: some terminals paint the rest of a line with an open background
/// colour when they scroll.
fn shaded(f: &mut fmt::Formatter<'_>, text: &str, colour: Rgb) -> fmt::Result {
    let Rgb(r, g, b) = colour;
    for line in text.split_inclusive('\n') {
        let content = line.trim_end_matches(['\r', '\n']);
        if !content.is_empty() {
            write!(f, "\x1b[30;48;2;{r};{g};{b}m{content}{RESET}")?;
        }
        f.write_str(&line[content.len()..])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::Probability;
    use crate::text;

    /// `rendered` without its colour codes.
    fn plain(rendered: &str) -> String {
        let mut out = String::new();
        let mut rest = rendered;
        while let Some(start) = rest.find("\x1b[") {
            out.push_str(&rest[..start]);
            let end = rest[start..].find('m').expect("terminated escape");
            rest = &rest[start + end + 1..];
        }
        out + rest
    }

    fn render(input: &str) -> String {
        let segments = text::segments(input);
        let report = Report::new(input, segments, |text| {
            Ok(Probability::from_logit(if text.contains("LLM") {
                9.0
            } else {
                -9.0
            }))
        })
        .unwrap();
        Heatmap::new(&report).to_string()
    }

    fn input() -> String {
        let prose = |marker: &str| format!("{marker} {}word.", "word ".repeat(120));
        format!(
            "{}\n\n```\ncode stays plain\n```\n\n{}\r\nwrapped\n",
            prose("LLM"),
            prose("human")
        )
    }

    #[test]
    fn colour_scale_runs_green_yellow_red() {
        assert_eq!(Rgb::shade(0.0), HUMAN);
        assert_eq!(Rgb::shade(0.5), EVEN);
        assert_eq!(Rgb::shade(1.0), LLM);
        assert_eq!(Rgb::shade(-1.0), HUMAN);
        assert_eq!(Rgb::shade(2.0), LLM);
    }

    #[test]
    fn rendering_reproduces_the_input_then_a_footer() {
        let input = input();
        let plain = plain(&render(&input));
        let (body, footer) = plain.split_at(input.len());
        assert_eq!(body, input);
        assert_eq!(footer, "\n50% overall · 0%            100%\n");
    }

    #[test]
    fn windows_are_shaded_by_score_and_code_is_left_plain() {
        let rendered = render(&input());
        let open = |Rgb(r, g, b): Rgb| format!("\x1b[30;48;2;{r};{g};{b}m");
        assert!(rendered.contains(&format!("{}LLM word", open(Rgb::shade(1.0)))));
        assert!(rendered.contains(&format!("{}human word", open(Rgb::shade(0.0)))));
        assert!(rendered.contains("\ncode stays plain\n"));
    }

    #[test]
    fn colour_closes_before_every_line_break() {
        let rendered = render(&input());
        for line in rendered.lines().filter(|line| line.contains("\x1b[30;")) {
            assert!(
                line.trim_end_matches('\r').ends_with(RESET) || line.ends_with(" 100%"),
                "{line:?}"
            );
        }
    }

    #[test]
    fn a_missing_final_newline_is_added() {
        let input = input();
        let input = input.trim_end();
        assert!(plain(&render(input)).starts_with(&format!("{input}\n\n")));
    }
}

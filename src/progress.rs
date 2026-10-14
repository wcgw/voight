//! Progress feedback on stderr. Nothing is drawn when stderr isn't a
//! terminal, so piped and scripted runs stay quiet; bars clear themselves
//! when dropped, including on error.

use std::borrow::Cow;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

/// How often bars redraw, so they keep moving during a long single step
/// such as the first GPU run compiling its kernels.
const TICK: Duration = Duration::from_millis(100);

const SPINNER: &str = "{spinner:.cyan} {msg} {elapsed:.dim}";
/// The message names the file being scored in a batch; `wide_msg` truncates
/// it to the terminal width, as a wrapped line would break the redraw.
const SCORING: &str =
    "{spinner:.cyan} Scoring {bar:30.cyan/blue} {pos}/{len} windows {elapsed:.dim} {wide_msg:.dim}";

/// A spinner showing `message` and the time elapsed.
pub fn spinner(message: impl Into<Cow<'static, str>>) -> ProgressBar {
    let bar = ProgressBar::new_spinner()
        .with_style(style(SPINNER))
        .with_message(message);
    bar.enable_steady_tick(TICK);
    bar
}

/// A bar counting `windows` as they are scored.
pub fn scoring(windows: usize) -> ProgressBar {
    let bar = ProgressBar::new(windows as u64).with_style(style(SCORING));
    bar.enable_steady_tick(TICK);
    bar
}

fn style(template: &str) -> ProgressStyle {
    // The templates above are constants, so this never falls back in
    // practice; a plain spinner beats a panic if one is ever malformed.
    ProgressStyle::with_template(template).unwrap_or_else(|_| ProgressStyle::default_spinner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_parse() {
        for template in [SPINNER, SCORING] {
            assert!(ProgressStyle::with_template(template).is_ok(), "{template}");
        }
    }
}

use std::fmt;

/// A probability in `[0, 1]`, displayed as a whole percentage.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f64);

impl Probability {
    /// Maps a log-odds value onto `[0, 1]` with the logistic function.
    pub fn from_logit(logit: f64) -> Self {
        Self(1.0 / (1.0 + (-logit).exp()))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    /// Mean of the probabilities weighted by their paired weight, or `None`
    /// when the weights sum to zero.
    pub fn weighted_mean(items: impl IntoIterator<Item = (Self, usize)>) -> Option<Self> {
        let (sum, total) = items
            .into_iter()
            .fold((0.0, 0), |(sum, total), (p, weight)| {
                (sum + p.0 * weight as f64, total + weight)
            });
        (total > 0).then(|| Self(sum / total as f64))
    }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `pad` so callers can align it with width specifiers.
        f.pad(&format!("{:.0}%", self.0 * 100.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_zero_is_even_odds() {
        assert_eq!(Probability::from_logit(0.0).to_string(), "50%");
    }

    #[test]
    fn extreme_logits_saturate() {
        assert_eq!(Probability::from_logit(1e6).to_string(), "100%");
        assert_eq!(Probability::from_logit(-1e6).to_string(), "0%");
    }

    #[test]
    fn weighted_mean_weighs_by_count() {
        let mean = Probability::weighted_mean([
            (Probability::from_logit(1e6), 3),
            (Probability::from_logit(-1e6), 1),
        ]);
        assert_eq!(mean.map(|p| p.to_string()).as_deref(), Some("75%"));
    }

    #[test]
    fn weighted_mean_of_nothing_is_none() {
        assert_eq!(Probability::weighted_mean([]), None);
        assert_eq!(
            Probability::weighted_mean([(Probability::from_logit(0.0), 0)]),
            None
        );
    }

    #[test]
    fn display_honours_width() {
        assert_eq!(format!("{:>5}", Probability::from_logit(0.0)), "  50%");
    }
}

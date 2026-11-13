//! Per-view triage: an average is not evidence that every evaluated camera is usable.
//! This does not identify floaters or certify unevaluated training/novel views.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct SuspectView {
    pub image: String,
    pub psnr: Option<f64>,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct ViewQuality {
    pub policy: &'static str,
    pub status: &'static str,
    pub scope: &'static str,
    pub n_evaluated: usize,
    pub median_psnr: Option<f64>,
    pub min_psnr: Option<f64>,
    pub suspect_views: Vec<SuspectView>,
}

pub fn view_quality<'a>(scores: impl IntoIterator<Item = (&'a str, Option<f64>)>) -> ViewQuality {
    let scores: Vec<_> = scores.into_iter().collect();
    let mut finite: Vec<_> = scores
        .iter()
        .filter_map(|(_, p)| p.filter(|p| p.is_finite()))
        .collect();
    finite.sort_by(f64::total_cmp);
    let median = (!finite.is_empty()).then(|| finite[finite.len() / 2]);
    let suspect_views: Vec<_> = scores
        .iter()
        .filter_map(|(name, p)| {
            let p = p.filter(|p| p.is_finite());
            let reason = match p {
                None => "missing_or_nonfinite_psnr",
                Some(p) if p < 10.0 => "psnr_below_10_db",
                Some(p) if finite.len() >= 8 && median.is_some_and(|m| m - p > 8.0) => {
                    "psnr_over_8_db_below_median"
                }
                _ => return None,
            };
            Some(SuspectView {
                image: (*name).into(),
                psnr: p,
                reason: reason.into(),
            })
        })
        .collect();
    ViewQuality {
        policy: "per-view-v1",
        status: if !suspect_views.is_empty() {
            "review_required"
        } else if scores.is_empty() {
            "insufficient"
        } else {
            "no_outliers_in_evaluated_views"
        },
        scope: "evaluated_views_only",
        n_evaluated: scores.len(),
        median_psnr: median,
        min_psnr: finite.first().copied(),
        suspect_views,
    }
}

impl ViewQuality {
    pub fn warning(&self) -> Option<String> {
        if self.suspect_views.is_empty() {
            return None;
        }
        let names: Vec<_> = self
            .suspect_views
            .iter()
            .take(8)
            .map(|v| v.image.as_str())
            .collect();
        Some(format!(
            "Per-view quality requires review: {} suspect views ({}). Inspect native/source comparisons; average scores do not establish every camera's quality. Unevaluated views remain unchecked.",
            self.suspect_views.len(),
            names.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restaurant_outlier_cannot_hide_in_good_average() {
        let quality = view_quality([
            ("frame_000222", Some(4.290630340576172)),
            ("a", Some(23.17)),
            ("b", Some(24.43)),
            ("c", Some(24.77)),
            ("d", Some(25.06)),
            ("e", Some(25.13)),
            ("f", Some(25.95)),
            ("g", Some(25.96)),
        ]);
        assert_eq!(quality.status, "review_required");
        assert_eq!(quality.suspect_views.len(), 1);
        assert_eq!(quality.suspect_views[0].image, "frame_000222");
        assert!(quality.warning().unwrap().contains("frame_000222"));
    }

    #[test]
    fn relative_failures_missing_metrics_and_sampling_scope_are_explicit() {
        let quality =
            view_quality((0..10).map(|i| ("capture", Some(if i == 3 { 19.0 } else { 30.0 }))));
        assert_eq!(quality.suspect_views.len(), 1);
        assert_eq!(
            quality.suspect_views[0].reason,
            "psnr_over_8_db_below_median"
        );
        assert_eq!(
            view_quality([("a", None), ("b", Some(f64::NAN))])
                .suspect_views
                .len(),
            2
        );
        assert_eq!(view_quality([]).status, "insufficient");
        let sampled = view_quality([
            ("rear120", Some(30.0)),
            ("rear180", Some(31.0)),
            ("rear240", Some(32.0)),
        ]);
        assert_eq!(sampled.status, "no_outliers_in_evaluated_views");
        assert_eq!(sampled.scope, "evaluated_views_only");
        assert!(sampled.warning().is_none());
    }
}

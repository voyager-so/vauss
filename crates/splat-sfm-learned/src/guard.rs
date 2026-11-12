//! Ground-truth-free guard after the learned fallback: the learned model replaces COLMAP's only
//! if it is not measurably worse, so the fallback cannot ship a model scoring below the one it
//! started from (ETH3D relief_2: the learned fallback scored AUC@3 .169 against COLMAP's .846 on
//! the same 31/31 frames).
//!
//! Candidate B (learned) replaces baseline A (COLMAP) only if every clause holds:
//! - registered(B) >= registered(A), counted over the same frame list;
//! - mean reprojection error(B) <= 1.10 * mean reprojection error(A);
//! - the weakest camera-graph cut of B (Fiedler sweep) has at least 0.25x the conductance of
//!   A's. On relief_2 the learned model was two camera groups joined by 4 shared-point pairs
//!   (conductance .0006 vs COLMAP's .15) with 19 cameras ~100 deg off, at a reprojection error
//!   only 1.06x COLMAP's;
//! - the held-out check ([`pick_by_heldout`]) picked learned. COLMAP is only a baseline when it
//!   is [`trainable`]; otherwise the learned model stands unguarded.
//!
//! Every decision, with the compared values of each clause, is written as JSON next to the
//! models (`guard-fallback.json`).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::colmap::{Cut, Model, read_model, weakest_cut};
use crate::{CameraSource, pick_by_heldout, trainable};

/// Allowed growth of the mean reprojection error.
pub const MAX_REPROJECTION_RATIO: f64 = 1.10;
/// Learned fallback: least fraction of COLMAP's weakest-cut conductance.
pub const MIN_CONDUCTANCE_RATIO: f64 = 0.25;

/// What a guard compares of one model.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelSummary {
    pub model_dir: PathBuf,
    /// Registered images whose name is in the frame list.
    pub registered: usize,
    /// Observations of 3D points.
    pub observations: usize,
    /// Mean Euclidean reprojection error over all observations, in pixels.
    pub mean_reprojection_error: f64,
}

/// One clause: the candidate's value against the baseline's and the bound it had to meet.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Clause {
    /// `registered`, `reprojection`, `connectivity` or `held_out`.
    pub name: &'static str,
    pub baseline: Option<f64>,
    pub candidate: Option<f64>,
    /// `candidate >= bound` for `registered`, `connectivity` and `held_out` (strictly greater),
    /// `candidate <= bound` otherwise.
    pub bound: Option<f64>,
    pub passed: bool,
    pub note: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardStep {
    /// Learned fallback (candidate) against COLMAP (baseline).
    Fallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kept {
    Baseline,
    Candidate,
}

/// A guard decision; also written to `guard-fallback.json`.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Guard {
    pub step: GuardStep,
    pub n_frames: usize,
    pub baseline: Option<ModelSummary>,
    pub candidate: Option<ModelSummary>,
    /// Empty when there was no baseline to guard against.
    pub clauses: Vec<Clause>,
    /// Clauses that rejected the candidate.
    pub fired: Vec<&'static str>,
    pub kept: Kept,
    /// The model to use from here on.
    pub model_dir: PathBuf,
    pub reason: String,
    pub failures: Vec<String>,
}

/// Guard the learned fallback (`learned`) against COLMAP's model (`colmap`, `None` if COLMAP made
/// none) on `frames` (the input frame names). `held_out` is the held-out check's mean PSNR
/// (COLMAP, learned) when it ran. Writes `out_dir/guard-fallback.json`.
pub fn guard_fallback(
    colmap: Option<&Path>,
    learned: &Path,
    frames: &[String],
    held_out: Option<(f64, f64)>,
    out_dir: &Path,
) -> Guard {
    let mut failures = Vec::new();
    let baseline = colmap.and_then(|dir| summarize(dir, frames, &mut failures));
    let candidate = summarize(learned, frames, &mut failures);
    let mut g = Guard {
        step: GuardStep::Fallback,
        n_frames: frames.len(),
        baseline: baseline.as_ref().map(|(s, _)| s.clone()),
        candidate: candidate.as_ref().map(|(s, _)| s.clone()),
        clauses: Vec::new(),
        fired: Vec::new(),
        kept: Kept::Candidate,
        model_dir: learned.to_path_buf(),
        reason: String::new(),
        failures,
    };
    match (&baseline, &candidate) {
        // Without a trainable baseline the learned model must still be trainable itself; a few
        // registered frames out of many are not a reconstruction (synthetic low-texture: 13/150, all wrong).
        (_, c) if !baseline.as_ref().is_some_and(|(a, _)| trainable(a.registered, frames.len()))
            && !c.as_ref().is_some_and(|(b, _)| trainable(b.registered, frames.len())) =>
        {
            g.kept = Kept::Baseline;
            g.model_dir = colmap.map(Path::to_path_buf).unwrap_or_default();
            g.reason = format!(
                "neither model is trainable (COLMAP {}, learned {} of {}): no usable fallback",
                baseline.as_ref().map_or("none".into(), |(a, _)| a.registered.to_string()),
                c.as_ref().map_or("none".into(), |(b, _)| b.registered.to_string()),
                frames.len()
            );
        }
        (None, _) => g.reason = "COLMAP made no readable model: no baseline, learned stands".into(),
        (Some((a, _)), _) if !trainable(a.registered, frames.len()) => {
            g.reason = format!("COLMAP is not trainable ({}/{} registered): no baseline, learned stands", a.registered, frames.len());
        }
        (Some((a, ma)), b) => {
            g.clauses = common_clauses(a, b.as_ref().map(|(s, _)| s));
            g.clauses.push(connectivity_clause(ma, b.as_ref().map(|(_, m)| m)));
            g.clauses.push(match held_out {
                Some((colmap_psnr, learned_psnr)) => Clause {
                    name: "held_out",
                    baseline: Some(colmap_psnr),
                    candidate: Some(learned_psnr),
                    bound: Some(colmap_psnr),
                    passed: pick_by_heldout(colmap_psnr, learned_psnr) == CameraSource::Learned,
                    note: "held-out PSNR on frames both registered; learned must be strictly better".into(),
                },
                None => Clause {
                    name: "held_out",
                    baseline: None,
                    candidate: None,
                    bound: None,
                    passed: false,
                    note: "COLMAP is trainable and the held-out check did not run: keep COLMAP".into(),
                },
            });
            decide(&mut g, &a.model_dir);
        }
    }
    write_json(&mut g, out_dir);
    g
}

/// The candidate's weakest-cut conductance must be at least [`MIN_CONDUCTANCE_RATIO`] times the
/// baseline's. Not applicable (passes) when the baseline graph is unsupported (fewer than six
/// cameras or an isolated camera); fails when only the candidate's is.
fn connectivity_clause(a: &Model, b: Option<&Model>) -> Clause {
    let (ca, cb) = (weakest_cut(a), b.and_then(weakest_cut));
    let describe = |c: &Option<Cut>| match c {
        Some(c) => format!("{} cameras on the small side, cross weight {}", c.names.len(), c.cross_weight),
        None => "unsupported graph".into(),
    };
    let bound = ca.as_ref().map(|c| c.conductance * MIN_CONDUCTANCE_RATIO);
    Clause {
        name: "connectivity",
        baseline: ca.as_ref().map(|c| c.conductance),
        candidate: cb.as_ref().map(|c| c.conductance),
        bound,
        passed: match (bound, &cb) {
            (None, _) => true,
            (Some(bound), Some(cb)) => cb.conductance >= bound,
            (Some(_), None) => false,
        },
        note: format!(
            "weakest-cut conductance, candidate >= {MIN_CONDUCTANCE_RATIO} x baseline{}; baseline: {}; candidate: {}",
            if ca.is_none() { " (not applicable: baseline graph unsupported)" } else { "" },
            describe(&ca),
            describe(&cb)
        ),
    }
}

/// The registered-count and reprojection clauses (failing when the candidate is unreadable).
fn common_clauses(a: &ModelSummary, b: Option<&ModelSummary>) -> Vec<Clause> {
    let bound = a.mean_reprojection_error * MAX_REPROJECTION_RATIO;
    vec![
        Clause {
            name: "registered",
            baseline: Some(a.registered as f64),
            candidate: b.map(|b| b.registered as f64),
            bound: Some(a.registered as f64),
            passed: b.is_some_and(|b| b.registered >= a.registered),
            note: "registered frames of the input list".into(),
        },
        Clause {
            name: "reprojection",
            baseline: Some(a.mean_reprojection_error),
            candidate: b.map(|b| b.mean_reprojection_error),
            bound: Some(bound),
            // NaN (no observations) fails.
            passed: b.is_some_and(|b| b.mean_reprojection_error <= bound),
            note: format!("mean reprojection error (px), candidate <= {MAX_REPROJECTION_RATIO} x baseline"),
        },
    ]
}

/// Keep the candidate (already in `g.model_dir`) only if no clause fired.
fn decide(g: &mut Guard, baseline: &Path) {
    g.fired = g.clauses.iter().filter(|c| !c.passed).map(|c| c.name).collect();
    if g.fired.is_empty() {
        g.reason = "every clause holds: candidate replaces baseline".into();
    } else {
        g.reason = format!("rejected by {}: keep baseline", g.fired.join(", "));
        g.kept = Kept::Baseline;
        g.model_dir = baseline.to_path_buf();
    }
}

fn summarize(dir: &Path, frames: &[String], failures: &mut Vec<String>) -> Option<(ModelSummary, Model)> {
    let m = read_model(dir).map_err(|e| failures.push(format!("reading {}: {e:#}", dir.display()))).ok()?;
    let frames: BTreeSet<&str> = frames.iter().map(String::as_str).collect();
    let registered = m.images.values().filter(|im| frames.contains(im.name.as_str())).count();
    let errors = m.squared_errors_by_name();
    let mean = errors.values().map(|e| e.sqrt()).sum::<f64>() / errors.len() as f64;
    let summary =
        ModelSummary { model_dir: dir.to_path_buf(), registered, observations: errors.len(), mean_reprojection_error: mean };
    Some((summary, m))
}

fn write_json(g: &mut Guard, out_dir: &Path) {
    let file = "guard-fallback.json";
    let json = serde_json::to_vec_pretty(g).map_err(anyhow::Error::from);
    if let Err(e) = json.and_then(|j| Ok(std::fs::write(out_dir.join(file), j)?)) {
        g.failures.push(format!("writing {file}: {e:#}"));
    }
}

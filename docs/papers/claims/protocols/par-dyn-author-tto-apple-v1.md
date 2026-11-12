# Approved first apple author-TTO pair — registration proposal

This is ONE descriptive seed pair, not a quality PASS or a full7 parity result.
The video sub-lead approved it at23:56PT10/08, ceiling$4/90min. Launch no earlier
than09:15PT10/09, after publications binds `par-dyn-author-tto-apple-v1` and its
immutable source/scorer SHA. The native numerical gate is a separate group,
`par-dyn-native-motion-fixture-v1`, with the already frozen1e-4 geometry limits.

## Arms and fair comparison

Both arms run consecutively inside one fal exec on one L40S, seed12345, complete
475 training/532 validation views, original2x pixels. The run reports paired
training/evaluation/total wall time and peak memory as well as descriptive quality.
The same-GPU timing comparison justifies `--gpu l40 --no-spill`; `--backend fal`
and detached collection prevent cancellation/overflow/resubmission. Timeout5400s
bounds billed L40 time near$3.11, leaving room under the$4 ceiling for setup.

- **Own**: original birth-time sparse DQ implementation with Apache gsplat1.5.3.
  30k static/30k dynamic initial primitives, final cap600k; up to1024 sampled
  observed dynamic tracks,16 neighbors, radius.1, global fusion from all train
  times, SH3. Geometry1500 steps, background3000, joint photometric8000. These are
  preregistered implementation defaults, chosen from paper/training conventions
  without validation selection. The exact executed source is the recipe.
- **Upstream**: MoSca@02404d2841b4e0dc2d9074860950bce5da579656, iPhone defaults.
  Recipe citations: `profile/iphone/iphone_fit.yaml:1-94` (LiDAR/BootsTAPIR,
  30k+30k,SH3,1500 geometry/8000 photo); `mosca_reconstruct.py:605-620`
  (implicit photo loss defaults); `mosca_evaluate.py:417-425` (iPhone TTO).
  The reference has no fixed primitive cap. Comparison mode is **both at their
  preregistered defaults**, not an equal-cap efficiency headline.

The own arm has direct lifted-node track supervision, one-level ARAP and bounded
clone/prune control. The upstream arm has its full multilevel graph, rendered
flow supervision, static/dynamic transfer and node control. Own gsplat raster
and upstream reference raster also differ. This compares complete implemented
recipes; it does not isolate a motion-kernel gain or assert recipe equivalence.
The upstream foundation-model preprocessing is frozen author data in BOTH arms;
reported training time does not include regenerating those priors.

## Camera and scorer contract

`author_tto.py` is an original implementation of the inspected algorithm contract
in AUTHOR-PROTOCOL.md. All scene parameters freeze after training; validation RGB
fits per-frame test cameras. Official co-visibility is used exclusively to score.
Train-only Umeyama similarity establishes the coordinate gauge. Supplied test
intrinsics remain fixed. Sequence order and initial/solved pose hashes are retained.

First frame per camera:300 Adam steps, quaternion/translation LR.003, final.0001.
Later frames:30 steps, previous solved pose,LR.0003,final.00001. Cosine decay starts
at step15. Fit negative PSNR on rendered alpha>.1. The scored image uses the last
pre-update transform; the solved pose hash uses the post-update transform.
Clip to[0,1] and round to uint8/255 in BOTH arms. Complete official2x masked PSNR,
masked partial-convolution SSIM and spatial Alex LPIPS; base metric source
`research/x6b/metrics.py` SHA256
`eb8a44f1d3ed53598028ad20c77751c9eb1e18d9fd6d84cbd954beb974cc1bd6`.
Canonical view-set hash (sorted names,no trailing newline):
`be449f8735336fc32ed49a43f07c1a101b155b33d723c5930aa9f65861adabf8`.
Official independently decoded-pixel manifest:
`$SPLAT_DATA/research/par-dyn/full7-eval-pixel-manifest.json`.
Fail if scored/unique views are not532. No fixed-camera rescore is in this job.

## Health, limits and interpretation

Both arms record dead fraction (opacity<.01) every100 optimizer steps, counts,
model size and the denominator. Own fixed-cap dead count flags above50% of600k.
Upstream default is uncapped: report cap=null and flag dead fraction>.5 of live
primitives explicitly. Retain every flagged row and check the upstream recipe;
never discard a collapsed competitor. No SfM arm exists: supplied cameras and
one frozen prior acquisition are shared, so rule:sfm-noise is not activated.

One seed cannot meet rule:seed-gate. Record the paired delta without PASS/AHEAD/
PARITY wording. A later quality claim requires a separately approved registration
with>=3 same-batch pairs, mean paired delta>2x sample SD, per-scene guard-.05dB and
Alex LPIPS guard+.002. No further seeds or jobs are authorized here.
Stop on nonfinite objective, invalid inputs, software adapter or timeout. An
exception or incomplete control remains a failed diagnostic, not a win. Preserve
rows and close every registry arm. Never cancel-and-resubmit. Any approved retry
of a DEAD request needs a fresh SPLAT_ATTEMPT; a dead collector is recovered.

## Prior art and portability

[MoSca](https://arxiv.org/html/2405.17421v2),CVPR2025: Eq.8-13 motivate observed
track lifting, source-time primitives, global fusion and ARAP. Its author MIT
scope excludes some raster/prior components; reference execution is research-only.
[Embedded Deformation](https://people.inf.ethz.ch/sumnerb/research/ed/),SIGGRAPH2007,
and [dual-quaternion skinning](https://users.cs.utah.edu/~ladislav/dq/index.html),
SIGGRAPH2007, precede this motion representation; no novelty claim is made.
Native `splat-train-motion` stays original and isolated. Author weights, external
source binaries and derived fixtures stay in remote research storage. No data or
trained benchmark weights ship and nothing is published externally.

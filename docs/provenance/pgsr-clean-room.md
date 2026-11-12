# PGSR: clean-room description

> Provenance record, kept because `crates/splat-train/src/multiview.rs` was implemented from this note and the
> PGSR paper only. Lightly edited for publication (internal team names and paths removed); the technical content
> is unchanged. "G1" below means splat's surfel training engine.

2026-10-09. **Purpose:** let G1 implement PGSR-style geometry without opening the PGSR repository.

**Sources:**
- Paper: Chen et al., *PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction*, [arXiv 2406.06521](https://arxiv.org/abs/2406.06521) (TVCG 2024).
- Reference code: `zju3dv/PGSR@de24f1a3` (2024-12-25). It was read for research only, under a **non-commercial** licence (plus the Inria 3DGS NC licence on the base files).

**Rules for this document:**
- Everything below is my own description of the algorithm. No code, identifiers or pseudocode were transcribed.
- Where the released code differs from the paper, both are stated. The DTU number we calibrate against comes from the code.
- Nothing from PGSR may enter the product. Implement from this note and the paper.

Measured calibration on our DTU-5 protocol is summarised in the table below (the raw report is not part of this repository).

## 1. The targets

| DTU Chamfer (mm) | 24 | 37 | 40 | 55 | 63 | 5-scan mean | 15-scan mean |
|---|---|---|---|---|---|---|---|
| PGSR, current arXiv / journal table (= README "Remove ICP" row) | 0.36 | 0.57 | 0.38 | 0.33 | 0.78 | **0.484** | 0.52 |
| PGSR, README "Paper" row | 0.34 | 0.58 | 0.29 | 0.29 | 0.78 | 0.456 | 0.53 |
| PGSR, README "Code_V1.0" (re-tuned, with ICP) | 0.33 | 0.51 | 0.29 | 0.28 | 0.75 | 0.432 | 0.47 |
| 2DGS (paper) | 0.48 | 0.91 | 0.39 | 0.39 | 1.01 | 0.636 | 0.80 |
| Official 2DGS on our harness (V-MESH-C v6, official scorer) | 0.499 | 0.804 | 0.354 | 0.406 | 0.946 | 0.602 | — |
| Old G1 build, g1-surfel-dtu5-v1 (V-MESH-C v6, official scorer; superseded) | 1.217 | 1.125 | 0.787 | 0.525 | 1.444 | 1.020 | — |
| G1 catch-up v1, arm C (current; pubs-verified, 2 seeds, official scorer) | — | — | — | — | — | 0.711 | — |

**ICP caveat:**
- The README's best row uses an ICP alignment step. ICP is not in the released evaluator.
- The released evaluator only applies the dataset scale matrix. So the comparable published number for a no-alignment protocol (ours) is the **"Remove ICP" row: 0.484 mm 5-scan mean**.

PGSR reports about 0.5 h of training per DTU scan, against about 0.3 h for 2DGS.

## 2. Representation: flattened 3D Gaussians, not 2D surfels

PGSR keeps **ordinary 3D Gaussians**:
- Rasterisation is the standard 3DGS EWA splat, including the 0.3 px screen-space dilation and front-to-back alpha blending, with the usual SH colour.
- Unlike 2DGS, there is no ray-splat intersection and no 2D disk primitive.

The Gaussians become planes only through a loss:
- **Flattening loss.** Each step, take every Gaussian visible in the current view. Find the smallest of its three activated scales and add `100 × mean(smallest scale)` to the loss. This drives one axis towards zero.
- **Per-Gaussian normal.** The normal is the rotation-matrix column belonging to the smallest scale axis. Its sign is flipped, per view, so it points towards the camera (positive dot product with the Gaussian-to-camera vector).
- **Per-Gaussian plane distance.** Transform the centre into camera space. The distance is the absolute dot product of the camera-space normal with the camera-space centre: the distance from the camera origin to the Gaussian's plane.

## 3. "Unbiased" plane depth (essential)

Alongside colour, the rasteriser alpha-blends four extra per-Gaussian channels with the same weights (alpha × transmittance):
- the camera-space normal (3 channels);
- the plane distance (1 channel);
- a constant 1, which gives accumulated opacity.

**Per-pixel depth** is the blended distance divided by the cosine between the blended normal and the pixel's ray. With the ray written as (x, y, 1) in normalised camera coordinates (pixel minus principal point, over focal length), depth = distance ÷ (−n · ray). The sign follows from normals facing the camera. The blended normal is **not** renormalised before the division.

The point is that the depth is the intersection of the ray with the blended *plane*, not a blend of centre depths. This avoids the curved, biased surfaces you get from alpha-blended centre depth. In 2DGS terms it plays the role of 2DGS's median or ray-plane depth.

**Gradients** flow into normals, distances and opacities through this division; the backward pass implements it analytically.

**Paper ablation** (TnT): F1 drops 0.52 → **0.38** without unbiased depth. This is one of the two largest single effects.

The same depth is used:
- in all geometric losses;
- for multi-view reprojection;
- for TSDF fusion at extraction.

## 4. Single-view geometric regularisation (normal consistency)

From step 7,000 on, every training step:
1. Back-project the plane-depth map to 3D points. For each pixel, take its four neighbours (up, down, left, right). Form the normal as the normalised cross product of the horizontal and vertical difference vectors. The one-pixel border gets no normal.
2. Multiply this **depth-derived normal** by the rendered accumulated opacity, detached (stop-gradient).
3. Loss: the per-pixel L1 distance (summed over xyz) between the depth-derived normal and the **rendered blended normal**, times an **edge-aware weight**, averaged over the image. Weight 0.015.

**Edge-aware weight:**
- On the ground-truth image, compute central differences in x and y, averaged over colour channels. Take the per-pixel max of the two, then min-max normalise over the image. The border is set to 1.
- The weight is (1 − normalised gradient)², clamped to [0, 1] and detached.
- Effect: the consistency constraint is relaxed at image edges, where depth discontinuities are likely.

**Paper ablation** (TnT):
- Without the single-view term: F1 0.52 → 0.49.
- Without only the edge weighting: 0.52 → 0.51.

**Moderate.** 2DGS already has a normal-consistency term (the largest effect in 2DGS's own ablation). G1 should already have an equivalent; the PGSR-specific part is the edge weighting, worth about 0.01 F1.

## 5. Multi-view regularisation (the decisive part for DTU)

Both multi-view terms start at **step 7,000** and use one neighbour per step.

**Neighbour graph,** computed once at load:
- For each training camera, rank the other cameras by centre distance, with optical-axis angle as the tiebreak.
- Keep those with angle below 30°, and centre distance between 0.01 and 1.5 scene units. Note the units: absolute in the dataset frame. DTU author data is normalised to roughly a unit sphere.
- Keep at most 8 per camera.
- Each step picks one of the current camera's neighbours uniformly at random. A camera with no neighbours skips the multi-view terms that step.

**Neighbour render:**
- Render the neighbour view with the same model, producing colour, normal, distance and plane depth.
- The neighbour's depth-derived normal is not needed.
- Gradients flow through the neighbour render as well.

### 5a. Geometric consistency: forward-backward reprojection

For every reference pixel:
1. Back-project it with the reference plane depth to a world point.
2. Project that point into the neighbour and bilinearly sample the **neighbour's plane depth** there.
3. Scale the neighbour-camera point along its ray to that sampled depth, giving the neighbour's surface point.
4. Transform it back into the reference camera and project to a pixel.
5. The **reprojection error φ** is the pixel distance between this round-trip pixel and the original pixel.

**Validity mask:**
- The projected point must land inside the neighbour image and have neighbour-camera z > 0.1.
- φ must be below **1 px**.

**Weight:** w = exp(−φ), detached, and zero where the mask fails.

**Loss:** the mean of w·φ over valid pixels, at weight **0.03**. Gradient reaches both depth maps through φ.

**Paper vs code:** the paper describes this term via a plane-induced homography. The released code does the round trip through the two rendered depth maps, as above. Both describe the same check at the surface.

### 5b. Photometric consistency: plane-induced homography NCC

Done on **grayscale** images (0.299 R + 0.587 G + 0.114 B).

**Resolution:**
- The DTU recipe sets the "NCC scale" to 0.5. The grayscale images used for NCC are therefore loaded at **2× the training resolution**. For DTU `-r2` that is the full 1554×1162 author crop, not the 777×581 training image.
- Pixel coordinates are scaled accordingly, and intrinsics are rescaled for the homography.

**Pixel sampling:**
- Only pixels that passed the geometric mask in 5a.
- At most **102,400** of them, chosen uniformly without replacement.
- Each sampled pixel keeps its weight w from 5a.

**Patch:** a **7×7** patch (half-size 3) around each sampled pixel in the reference grayscale. The reference patch values are constants.

**Homography for each pixel:**
- Built from the reference pixel's rendered blended normal n and rendered blended distance d.
- With relative rotation R and translation t from reference to neighbour, H = K_nei (R − t nᵀ / d) K_ref⁻¹. Use the intrinsics at the NCC resolution.
- Warp the patch's pixel coordinates through H, then bilinearly sample the **neighbour's ground-truth grayscale**.
- Gradients flow **only** into the rendered normal and distance of the reference view, through H. No gradient flows into colours.

**NCC score:**
- Over the 49 samples: squared correlation, cc = cov² / (var_ref · var_nei + 1e-8).
- Score = 1 − cc, clamped to [0, 2].
- Patches with score ≥ 0.9 are discarded: textureless or badly mismatched.
- Loss: the mean of w × score over the kept patches, at weight **0.15**.

**Paper ablation** (TnT F1, full model 0.52):
- Without **both** multi-view terms: **0.32**. This is the largest effect in the paper.
- Without photometric only: 0.39.
- Without geometric only: 0.49.
- Without the occlusion-aware weighting (w = 1 on the raw mask): **0.28**, and PSNR collapses 26.7 → 21.7.

The forward-backward weighting is essential, not a nicety: without it, the photometric term pulls occluded pixels into wrong geometry.

**Applied to 2DGS** (paper Table I3, TnT F1):

| Variant | F1 |
|---|---|
| 2DGS | 0.32 |
| 2DGS + multi-view | 0.41 |
| 2DGS with expected (blended) depth + multi-view | 0.47 |

This is the most direct evidence for G1: the multi-view terms port onto a surfel model and give most of the gain.

## 6. Exposure compensation (not used on DTU)

**Mechanism:**
- Each image gets two scalars (a, b), initialised to 0 with Adam learning rate 1e-3. Rendered colour becomes exp(a)·I + b.
- Turned on after step 1,000.
- The compensated image feeds only the L1 part of the colour loss, and only while 1 − SSIM < 0.5. SSIM always uses the raw render.
- The meshing render uses the uncompensated image.

**Where it is used:** the released DTU recipe leaves it **off**. The paper uses it for Tanks and Temples only.

**Ablation** (TnT): F1 0.52 → 0.49, PSNR 26.7 → 25.3.

**For DTU parity: skip.** For phone video it is relevant (it is a per-image affine colour, like our appearance work).

## 7. Densification and pruning (incidental to the geometry idea, but it affects results)

**Base:** 3DGS adaptive density control.
- Every 100 steps, from step 500 up to 15,000.
- Gradient threshold 2e-4. Clone if the largest scale ≤ 0.001 × scene extent; split otherwise.
- **Splitting:** 2 children, sampled from the parent Gaussian. Scales are divided by **1.6**, the same as 3DGS.
- Opacity reset every 3,000 steps.
- **Pruning:**
  - opacity below **0.005**;
  - from step 3,000, screen radius over 20 px;
  - world-space largest scale over 0.1 × extent.

**AbsGS addition:**
- A second accumulator holds the norm of the per-pixel **absolute** screen-space gradients, which do not cancel across pixels.
- Gaussians not already selected for split are also split if:
  - the absolute-gradient average is ≥ **8e-4**;
  - their largest scale exceeds the clone limit;
  - their max screen radius exceeds **20 px**.
- At most **50,000** absolute-gradient splits per densification step, keeping the top ones by quantile.
- Global cap of 6M Gaussians.

**Multi-view trim:**
- Every 1,000 steps before step 15,000, render all training views.
- A Gaussian counts as *observed* in a view if it contributes to some pixel while the transmittance in front of it is still above 0.5.
- Prune Gaussians observed in **fewer than 2** views.
- This removes single-view floaters, which matter for TSDF.

**Other settings:**

| Setting | Value |
|---|---|
| Steps | 30,000 |
| Learning rates | Standard 3DGS (position 1.6e-4 → 1.6e-6 exponential decay, scaled by extent; SH DC 2.5e-3, rest 1/20; opacity 0.05; scale 5e-3; rotation 1e-3) |
| SH degree | Increases every 1,000 steps to 3 |
| Background | Black |
| Colour loss | 0.8 L1 + 0.2 (1 − SSIM) |
| RNG seed | Fixed |

## 8. Total loss and schedule

| Steps | Loss |
|---|---|
| 1 to 7,000 | colour + 100 × flattening |
| After 7,000 | + 0.015 × single-view normal + 0.03 × multi-view geometric + 0.15 × multi-view NCC |

Cost after 7k: each step renders one extra view (the neighbour), plus NCC on up to 102k patches. That is why PGSR runs about 0.5 h against about 0.3 h for 2DGS.

## 9. Mesh extraction (the DTU recipe)

1. Render plane depth for all **training** views at the training resolution.
2. **Foreground mask:** the author's preprocessed DTU PNGs carry an alpha mask. PGSR (like official 2DGS) trains on the full RGB, background included, but **zeroes depth where alpha < 0.5** before fusion. This is a protocol choice: it uses dataset masks at extraction time.
3. Zero depth beyond **5.0** units.
4. Fuse with a scalable TSDF:
   - voxel **0.002** in the normalised frame (half of 2DGS's 0.004);
   - truncation **4 voxels** (0.008);
   - colour from the render;
   - marching cubes.
5. Keep only the **largest connected component** (and drop components below 50 triangles).

There is no smoothing, decimation or carving.

**Optional depth filter:** zero pixels whose depth-derived normal is more than 80° from the view ray. It is **off** for DTU and on for TnT, where it is worth about 0.02 F1.

**Evaluation:**
- The mesh is mapped to DTU millimetres with the dataset scale matrix, then scored with the standard DTU Python evaluator.
- PGSR's own scoring script **does not** silhouette-cull the mesh with masks. Official 2DGS's script does (24-px dilated masks).
- We report both, plus B15.

## 10. What is essential vs incidental for G1

| Component | Status | Evidence | Note for G1 (surfel engine) |
|---|---|---|---|
| Multi-view photometric NCC with plane homography | **Essential** | −0.13 F1 when removed (TnT) | Needs a neighbour render per step and a grayscale patch sampler. The homography uses the blended normal and distance, which G1's surfels already have. |
| Forward-backward occlusion weight w | **Essential** (do not ship NCC without it) | −0.24 F1, PSNR −5 dB when removed | Needs the neighbour depth map; cheap once the neighbour render exists. |
| Plane ("unbiased") depth | **Essential** | −0.14 F1 | G1's ray-plane depth is the surfel analogue. Make sure TSDF and the multi-view losses use it, not centre depth. |
| Multi-view geometric loss | Helpful | −0.03 F1 | Comes almost free with w. |
| Single-view normal consistency + edge weight | Helpful | −0.03 F1 (edge weight −0.01) | G1 likely has the 2DGS version; add the (1 − grad)² weight. |
| Flattening loss | Needed for 3D Gaussians only | — | Not needed with true 2D surfels. |
| AbsGS densification + multi-view trim | Incidental but real | Not ablated | The trim (seen in < 2 views) is cheap and helps TSDF floaters. |
| Exposure compensation | Not on DTU | −0.03 F1 on TnT | Relevant for phone video later. |
| TSDF voxel 0.002 + largest component + masked fusion | Protocol | Not ablated | Match these before comparing numbers (see `2dgs-recipe-diff.md` §1). |

**Recommended G1 order:**
1. Plane depth everywhere: render, losses, TSDF.
2. Neighbour graph + neighbour render + forward-backward w + geometric loss (cheap; validates the plumbing).
3. NCC photometric term at 2× resolution with 7×7 patches.
4. Edge-weighted normal loss + multi-view trim.

Expected outcome:
- The paper's own 2DGS + multi-view variant closed most of the PGSR–2DGS gap on TnT (0.32 → 0.47 against PGSR's 0.52).
- On DTU the 2DGS → PGSR gap is 0.636 → 0.484 mm (5-scan, published).

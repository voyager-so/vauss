# splat-inpaint

Object removal + hole fill on a trained splat, in-engine (Rust, Burn/wgpu + ONNX Runtime; no
Python in the shipped path). Port of a Python (gsplat) reference implementation; algorithm notes
are in `src/lib.rs`.

- Entry points: `splat_inpaint::run(InpaintJob) -> InpaintResult` and
  `splat_inpaint::run_job_file(job.json)`; the backend runs it as the detached worker
  `splat __brush inpaint <job.json>` (crates/splat-brush/src/inpaint.rs).
- Models (Apache-2.0, pinned in `crates/splat-fetch/assets.json`): `inpaint-lama`
  (Carve/LaMa-ONNX `lama_fp32.onnx` @ c3c0c9e4) and `inpaint-depth-anything-v2-small`
  (onnx-community/depth-anything-v2-small `onnx/model.onnx` @ 4472b736, saved as
  `depth_anything_v2_small.onnx`). Looked up in the job's `models_dir`, then
  `$SPLAT_MODELS_DIR`, else downloaded into the splat cache (SHA-256 checked). Both run on the
  ORT CPU provider (LaMa 512x512 and DA at ~518 px take seconds).
  Diffusion inpainter (`"inpainter": "sd15"` in job.json; knobs `prompt`, `negative_prompt`,
  `diffusion_steps` (30), `guidance` (7.5), `seed`): SD-1.5 inpainting, fp16 diffusers ONNX export
  `RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16` @ a612b433 (CreativeML OpenRAIL-M; assets
  `inpaint-sd15-*`, ~2.1 GB), in `src/diffusion.rs`: own CLIP BPE tokenizer, DDIM (eta 0, leading
  spacing, from scheduler_config.json), CFG batch 2, seeded latent noise. Runs on ORT CUDA when
  the runtime + CUDA 12/cuDNN 9 libraries resolve ($SPLAT_INPAINT_ORT_CUDA=0/1 forces CPU/CUDA),
  else CPU with a warning. result.json: `inpainter`, `provider`, `timings_s.diffusion_s` and
  `diffusion_{load,text,vae_encode,unet,vae_decode}_s`. ONNX signatures (read from the graphs):
  text `input_ids` int32 [b,77] -> `last_hidden_state` f16 [b,77,768]; UNet `sample` f16
  [b,9,64,64], `timestep` f16 [b], `encoder_hidden_states` f16 [b,77,768] -> `out_sample`
  [b,4,64,64]; VAE enc `sample` f16 [-1,1] -> `latent_sample` (posterior SAMPLE, unseeded
  RandomNormalLike: tiny run-to-run jitter); VAE dec `latent_sample` -> `sample` in [-1,1].
  Verified input metadata: LaMa `image` [b,3,512,512] in [0,1] + `mask` [b,1,512,512] (1 = hole),
  `output` in 0..255; DA `pixel_values` [b,3,H,W] (multiples of 14) -> `predicted_depth` [b,H,W].
- Trainer hook (crates/splat-train): `TrainConfig::freeze_prefix` restores the first K1 splats
  after every step; `freeze_geometry_prefix` K2 additionally restores only the geometry of rows
  K1..K2 (their SH + opacity train). The fine-tune orders splats [frozen old, recolourable old
  (`recolour`, `recolour_pad`: kept old splats in the reference hole inside the removed set's 3D
  box padded 30%), new]; recoloured records are rewritten in place (SH + opacity only); `qat_tail = false` skips the quantisation tail. The fine-tune also turns off
  robust masking, background noise, the opacity regulariser and mean noise, never calls refine,
  and trains at full resolution from step 0.

Differences from the Python reference:
- Fine-tune trains all SH bands of the new splats (Python: DC only) with the trainer's default
  LRs and SSIM weight 0.2; the reference view is repeated in the loader so it is ~1/`ref_every`
  of the batches instead of every `ref_every`-th step.
- Unprojection and pruning use a pinhole approximation of the capture camera (lens distortion
  dropped); renders use the Brush camera as trained.
- No consistency (reprojected-fill) metrics or held-out grid yet; result.json has held-out PSNR
  outside the hole (edited vs unedited render, and removed vs unedited) plus ref-view
  before/hole/removed/fill/after PNGs and a strip.

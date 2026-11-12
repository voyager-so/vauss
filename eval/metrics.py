#!/usr/bin/env python3
"""Image-quality metrics between two folders of images (ground truth vs renders).

Usage:
    python eval/metrics.py GT_DIR RENDER_DIR [--json out.json] [--mask-dir M] [--device cuda]
                                              [--no-lpips] [--no-cc] [--resize]

Images are paired by file stem (gt/00012.png <-> renders/00012.jpg). Every metric is
computed per image on float RGB in [0,1] and then averaged (the 3DGS / gsplat convention).

Metrics:
  psnr        10*log10(1/MSE), data_range=1
  ssim        Gaussian window 11, sigma 1.5 (torchmetrics; same as the 3DGS repo's ssim())
  lpips_vgg   LPIPS VGG16 backbone (the number the 3DGS paper reports)
  lpips_alex  LPIPS AlexNet backbone (the number gsplat's benchmark reports)
  cc_*        the same metrics after per-image colour correction of the render onto the GT.
              Default --cc quadratic is the Zip-NeRF eval colour correction (Barron et al. 2023,
              camp_zipnerf image_utils.color_correct, Apache-2.0): per-channel least squares over a
              quadratic+linear+bias expansion, 5 iterations, ignoring clipped pixels. Spirula's native
              cc_* metrics use the same algorithm. --cc affine is a plain 3x4 fit. This removes global
              exposure/white-balance drift, which matters for phone video with auto exposure.
              Report it separately; never as the headline.

Spirula run dirs: `metrics.py RUN_DIR` with no RENDER_DIR pairs eval-gt-N.png with eval-render-N.png
(what `spirula train --save-eval-images 1` writes). Note Spirula's own lpips_vgg uses
normalize=False on [0,1] input (reference/python/README.md upstream), so its lpips_vgg is not
comparable with ours or with published numbers; ours uses normalize=True for both nets.

With --mask-dir, masks (white = keep, same stem as GT) zero-out ignored pixels in both images
before scoring and PSNR is computed over kept pixels only. SSIM/LPIPS are still full-frame on the
masked images.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def list_images(d: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS}


def load_rgb(p: Path, size=None) -> torch.Tensor:
    im = Image.open(p).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_mask(p: Path, size) -> torch.Tensor:
    im = Image.open(p).convert("L")
    if im.size != size:
        im = im.resize(size, Image.NEAREST)
    return torch.from_numpy(np.asarray(im) >= 128).float()[None]


def color_correct_quadratic(render: torch.Tensor, gt: torch.Tensor, mask=None,
                            num_iters: int = 5, eps: float = 0.5 / 255) -> torch.Tensor:
    """Zip-NeRF style colour correction (see module docstring). render, gt: 3xHxW in [0,1]."""
    cur = render.reshape(3, -1).T.double()
    ref = gt.reshape(3, -1).T.double()
    keep = torch.ones(cur.shape[0], dtype=torch.bool, device=cur.device) if mask is None else mask.reshape(-1) > 0

    def unclipped(z):
        return (z >= eps) & (z <= 1 - eps)

    mask0 = unclipped(cur)
    for _ in range(num_iters):
        a = torch.cat([cur[:, i:i + 1] * cur[:, i:] for i in range(3)] + [cur, torch.ones_like(cur[:, :1])], -1)
        warp = []
        for i in range(3):
            m = (mask0[:, i] & unclipped(cur[:, i]) & unclipped(ref[:, i]) & keep).double()
            ma = a * m[:, None]
            warp.append(torch.linalg.lstsq(ma.T @ ma, ma.T @ (ref[:, i] * m)).solution)
        cur = (a @ torch.stack(warp, -1)).clamp(0, 1)
    return cur.T.reshape(render.shape).float()


def color_correct_affine(render: torch.Tensor, gt: torch.Tensor, mask=None) -> torch.Tensor:
    """Affine colour transform fitted by least squares so that A @ [r,g,b,1] ~= gt."""
    c, h, w = render.shape
    x = torch.cat([render.reshape(3, -1), torch.ones(1, h * w, device=render.device)], 0).T
    y = gt.reshape(3, -1).T
    if mask is not None:
        keep = mask.reshape(-1) > 0
        x, y = x[keep], y[keep]
    a = torch.linalg.lstsq(x.double(), y.double()).solution.float()  # 4x3
    xa = torch.cat([render.reshape(3, -1), torch.ones(1, h * w, device=render.device)], 0).T
    return (xa @ a).T.reshape(3, h, w).clamp(0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt_dir", type=Path)
    ap.add_argument("render_dir", type=Path, nargs="?", help="omit for a Spirula run dir (eval-gt-*/eval-render-*)")
    ap.add_argument("--json", type=Path, help="write per-image + mean metrics here")
    ap.add_argument("--mask-dir", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-cc", action="store_true", help="skip colour-corrected variants")
    ap.add_argument("--cc", choices=["quadratic", "affine"], default="quadratic")
    ap.add_argument("--resize", action="store_true", help="resize renders to GT size instead of failing")
    args = ap.parse_args()

    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

    if args.render_dir is None:
        gts = {p.stem[len("eval-gt-"):]: p for p in sorted(args.gt_dir.glob("eval-gt-*.png"))}
        rds = {p.stem[len("eval-render-"):]: p for p in sorted(args.gt_dir.glob("eval-render-*.png"))}
    else:
        gts, rds = list_images(args.gt_dir), list_images(args.render_dir)
    stems = sorted(set(gts) & set(rds))
    missing = sorted(set(gts) - set(rds))
    if not stems:
        print(f"no paired images between {args.gt_dir} and {args.render_dir}", file=sys.stderr)
        return 2
    if missing:
        print(f"warning: {len(missing)} GT images have no render (e.g. {missing[:3]})", file=sys.stderr)

    dev = torch.device(args.device)
    lp = {}
    if not args.no_lpips:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
        # normalize=True: inputs are in [0,1]; torchmetrics rescales to [-1,1] internally.
        lp = {n: LPIPS(net_type=n, normalize=True).to(dev).eval() for n in ("vgg", "alex")}

    t0 = time.time()
    rows = []
    for s in stems:
        gt = load_rgb(gts[s]).to(dev)
        size = (gt.shape[2], gt.shape[1])
        rimg = Image.open(rds[s])
        if rimg.size != size and not args.resize:
            print(f"size mismatch for {s}: gt {size} render {rimg.size} (use --resize)", file=sys.stderr)
            return 2
        rd = load_rgb(rds[s], size).to(dev)
        m = None
        if args.mask_dir:
            mp = next((p for p in args.mask_dir.glob(s + ".*")), None)
            if mp is not None:
                m = load_mask(mp, size).to(dev)
        variants = {"": rd}
        if not args.no_cc:
            cc = color_correct_quadratic if args.cc == "quadratic" else color_correct_affine
            variants["cc_"] = cc(rd, gt, m)
        row = {"image": s}
        for pre, r in variants.items():
            g = gt
            if m is not None:
                r, g = r * m, gt * m
                mse = ((r - g) ** 2).sum() / (m.sum() * 3).clamp(min=1)
            else:
                mse = ((r - g) ** 2).mean()
            row[pre + "psnr"] = float(-10 * torch.log10(mse.clamp(min=1e-10)))
            with torch.no_grad():
                row[pre + "ssim"] = float(ssim_fn(r[None], g[None], data_range=1.0, gaussian_kernel=True,
                                                  kernel_size=11, sigma=1.5))
                for n, model in lp.items():
                    row[f"{pre}lpips_{n}"] = float(model(r[None], g[None]))
        rows.append(row)

    keys = [k for k in rows[0] if k != "image"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out = {"gt_dir": str(args.gt_dir), "render_dir": str(args.render_dir or args.gt_dir), "cc": args.cc, "n": len(rows),
           "missing_renders": missing, "device": str(dev), "seconds": round(time.time() - t0, 2),
           "mean": mean, "per_image": rows}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
    print(f"n={len(rows)}  " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

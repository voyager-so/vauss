"""Cheap capture-quality statistics, shared by the CLI and pipeline backends."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def summarize_frames(frames: np.ndarray) -> dict:
    """Summarize N x H x W grayscale samples in the range [0, 255]."""
    fr = np.asarray(frames, dtype=np.float32)
    if fr.ndim != 3 or not len(fr) or min(fr.shape[1:]) < 3:
        raise ValueError("Expected at least one grayscale frame of size 3x3 or larger")
    lap = np.array([(x[1:-1, :-2] + x[1:-1, 2:] + x[:-2, 1:-1] + x[2:, 1:-1]
                    - 4 * x[1:-1, 1:-1]).var() for x in fr])
    lum = fr.mean((1, 2))
    diff = np.abs(np.diff(fr, axis=0)).mean((1, 2))
    median, p10 = float(np.median(lap)), float(np.percentile(lap, 10))
    return {
        "sample_count": len(fr), "sharpness_median": median, "sharpness_p10": p10,
        "sharpness_p10_over_median": p10 / max(median, 1e-6),
        "luminance_min": float(lum.min()), "luminance_max": float(lum.max()),
        "luminance_range": float(lum.max() - lum.min()),
        "black_frac": float(((fr < 16).mean((1, 2)) > 0.98).mean()),
        "top_clipped_fraction": float((fr[:, :fr.shape[1] // 3] >= 236).mean()),
        "static_fraction": float((diff < 1).mean()) if len(diff) else None,
    }


def sample_video(path: str | Path, fps: float = 4, max_frames: int | None = None) -> np.ndarray:
    if fps <= 0 or (max_frames is not None and max_frames < 1):
        raise ValueError("fps and max_frames must be positive")
    argv = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale=640:360"]
    if max_frames is not None:
        argv += ["-frames:v", str(max_frames)]
    raw = subprocess.run(argv + ["-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 360, 640)


def inspect_video(path: str | Path, fps: float = 4, max_frames: int | None = None) -> dict:
    return summarize_frames(sample_video(path, fps, max_frames))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--fps", type=float, default=4)
    args = ap.parse_args()
    for path in args.videos:
        s = inspect_video(path, args.fps)
        static = f"{s['static_fraction']:.2f}" if s['static_fraction'] is not None else "unknown"
        print(f"{path}: n={s['sample_count']} sharp med={s['sharpness_median']:.0f} "
              f"p10={s['sharpness_p10']:.0f} (p10/med {s['sharpness_p10_over_median']:.2f}) | "
              f"lum {s['luminance_min']:.0f}-{s['luminance_max']:.0f} | "
              f"top clipped {100 * s['top_clipped_fraction']:.1f}% | static frac {static}")


if __name__ == "__main__":
    main()

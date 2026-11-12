#!/usr/bin/env python3
"""Render the tiny synthetic proof scene in testdata/synth12/ (numpy + Pillow only).

A textured room (floor + four walls) holding three textured boxes, seen by N pinhole
cameras on an arc. Every surface carries multi-scale colour value noise so SIFT finds
plenty of features even at 320 px. Deterministic: the same seed renders the same bytes.
We render our own scene so the committed test data has no third-party licence.

usage: make_synthetic_scene.py OUT_DIR [--views 12] [--width 320] [--height 240]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOM = (np.array([-4.0, -4.0, 0.0]), np.array([4.0, 4.0, 3.5]))
BOXES = [
    (np.array([-1.2, -0.6, 0.0]), np.array([-0.2, 0.4, 1.1])),
    (np.array([0.3, -1.0, 0.0]), np.array([1.1, -0.2, 0.6])),
    (np.array([0.2, 0.5, 0.0]), np.array([0.9, 1.2, 1.6])),
]


def value_noise(seed, u, v, scale):
    """Bilinear value noise on a lattice of `scale` world units, one RGB colour per node."""
    rng = np.random.default_rng(seed)
    lattice = rng.random((97, 97, 3))
    x, y = u / scale, v / scale
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    at = lambda i, j: lattice[i % 97, j % 97]
    top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
    bottom = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
    return top * (1 - fy) + bottom * fy


def texture(surface, u, v):
    colour = sum(w * value_noise(surface * 10 + k, u, v, s) for k, (w, s) in enumerate([(0.5, 0.6), (0.3, 0.17), (0.2, 0.05)]))
    # Hard-edged blobs give SIFT stable corners on top of the smooth noise.
    spots = value_noise(surface * 10 + 7, u, v, 0.25)[..., 0] > 0.72
    colour = np.where(spots[..., None], 1.0 - colour, colour)
    return np.clip(colour, 0.0, 1.0)


def slab(origin, direction, lo, hi):
    """Ray/AABB entry and exit distances plus the axis hit on entry and exit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - origin) / direction
        t2 = (hi - origin) / direction
    near, far = np.minimum(t1, t2), np.maximum(t1, t2)
    return near.max(-1), far.min(-1), near.argmax(-1), far.argmin(-1)


def render(eye, target, width, height, fov_deg=60.0, supersample=2):
    w, h = width * supersample, height * supersample
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    xs, ys = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    dirs = forward * f + right * xs[..., None] - up * ys[..., None]
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    eye_b = np.broadcast_to(eye, dirs.shape)
    # Inside the room every ray exits through exactly one face.
    _, t_room, _, axis_room = slab(eye_b, dirs, *ROOM)
    best_t, axis, surface = t_room, axis_room, np.zeros(dirs.shape[:2], int)
    for i, (lo, hi) in enumerate(BOXES, start=1):
        t_in, t_out, a_in, _ = slab(eye_b, dirs, lo, hi)
        hit = (t_in <= t_out) & (t_in > 1e-4) & (t_in < best_t)
        best_t = np.where(hit, t_in, best_t)
        axis = np.where(hit, a_in, axis)
        surface = np.where(hit, i, surface)
    p = eye + dirs * best_t[..., None]
    # Texture coordinates: the two world axes spanning the face; distinct seeds per face.
    u = np.choose(axis, [p[..., 1], p[..., 0], p[..., 0]]) + 10
    v = np.choose(axis, [p[..., 2], p[..., 2], p[..., 1]]) + 10
    face = surface * 3 + axis
    colour = np.zeros(dirs.shape)
    for fid in np.unique(face):
        m = face == fid
        colour[m] = texture(int(fid), u[m], v[m])
    normal = np.eye(3)[axis]
    shade = 0.55 + 0.45 * np.abs(normal @ np.array([0.3, 0.5, 0.81]))
    img = colour * shade[..., None]
    img = img.reshape(height, supersample, width, supersample, 3).mean((1, 3))
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = np.array([0.0, 0.0, 0.6])
    for i in range(args.views):
        # A 100-degree arc at 3 m radius: neighbouring views overlap heavily.
        a = np.radians(-50 + 100 * i / max(args.views - 1, 1))
        eye = np.array([3.0 * np.cos(a), 3.0 * np.sin(a), 1.6 + 0.2 * np.sin(3 * a)])
        Image.fromarray(render(eye, target, args.width, args.height)).save(args.out / f"{i + 1:06d}.jpg", quality=92)


if __name__ == "__main__":
    main()

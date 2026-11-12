#!/usr/bin/env python3
"""Export LPIPS v0.1 (VGG16 and AlexNet) to ONNX for splat-eval's Rust LPIPS (crates/splat-eval/src/lpips.rs).

Dev-time only: the Rust crate needs just the .onnx files at runtime. Run with the eval venv
(torch, torchvision, torchmetrics) plus `onnx` (the exporter needs it) and `onnxruntime` (self-check)
importable, e.g. from a side dir so the shared venv stays untouched:

    uv pip install --python $PY --target /tmp/onnxdeps onnx onnxruntime   # then drop its numpy copy
    CUDA_VISIBLE_DEVICES= PYTHONPATH=/tmp/onnxdeps $PY crates/splat-eval/tools/export_lpips_onnx.py \
        --out models [--check-pair GT.png RENDER.png]

With torch 2.14.1 / torchmetrics 1.9.0 the output is byte-identical across runs (sha256 in models.json).

The exported graph is torchmetrics' `_LPIPS` (the same module `LearnedPerceptualImagePatchSimilarity`
builds: richzhang LPIPS v0.1 linear weights bundled with torchmetrics, torchvision ImageNet backbones)
called with normalize=False:

    inputs  in0, in1: float32 [1, 3, H, W], already in [-1, 1]   (H, W dynamic)
    output  distance: float32 [1, 1, 1, 1]

torchmetrics' normalize=True (inputs in [0,1]) is just `2*x - 1` before this graph; splat-eval does
that in Rust so the graph stays the plain LPIPS definition. The ScalingLayer (ImageNet shift/scale)
is inside the graph.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

OPSET = 17
NETS = {"vgg": "lpips_vgg.onnx", "alex": "lpips_alex.onnx"}


class LpipsDistance(nn.Module):
    def __init__(self, net: str):
        super().__init__()
        from torchmetrics.functional.image.lpips import _NoTrainLpips

        self.lpips = _NoTrainLpips(net=net).eval()

    def forward(self, in0: torch.Tensor, in1: torch.Tensor) -> torch.Tensor:
        return self.lpips(in0, in1, normalize=False)


def export(net: str, path: Path) -> None:
    model = LpipsDistance(net)
    dummy = (torch.rand(1, 3, 96, 128) * 2 - 1, torch.rand(1, 3, 96, 128) * 2 - 1)
    # TorchScript exporter: self-contained single file, no onnxscript needed, stable op choice.
    torch.onnx.export(
        model,
        dummy,
        str(path),
        dynamo=False,
        opset_version=OPSET,
        input_names=["in0", "in1"],
        output_names=["distance"],
        dynamic_axes={"in0": {2: "height", 3: "width"}, "in1": {2: "height", 3: "width"}},
        do_constant_folding=True,
    )


def load01(p: Path) -> torch.Tensor:
    from PIL import Image

    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def check(net: str, path: Path, pair: tuple[Path, Path] | None) -> float:
    """Max |torchmetrics(normalize=True) - ORT(2x-1)| over random and (optionally) real inputs, on CPU."""
    import onnx
    import onnxruntime as ort
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    onnx.checker.check_model(str(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ref = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).eval()
    g = torch.Generator().manual_seed(0)
    cases = [(torch.rand(1, 3, h, w, generator=g), torch.rand(1, 3, h, w, generator=g))
             for h, w in [(64, 64), (97, 131), (546, 979)]]
    if pair is not None:
        cases.append((load01(pair[0]), load01(pair[1])))
    worst = 0.0
    for a, b in cases:
        want = float(ref(a, b))
        ref.reset()
        got = float(sess.run(None, {"in0": (2 * a - 1).numpy(), "in1": (2 * b - 1).numpy()})[0].reshape(()))
        worst = max(worst, abs(want - got))
        print(f"  {net} {tuple(a.shape[2:])}: torchmetrics {want:.7f} ort {got:.7f}", file=sys.stderr)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nets", nargs="+", default=list(NETS), choices=list(NETS))
    ap.add_argument("--check-pair", nargs=2, type=Path, metavar=("GT", "RENDER"))
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torchmetrics

    info = {}
    for net in args.nets:
        path = args.out / NETS[net]
        export(net, path)
        data = path.read_bytes()
        info[net] = {
            "file": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "torch": torch.__version__,
            "torchmetrics": torchmetrics.__version__,
            "opset": OPSET,
        }
        if not args.no_check:
            info[net]["max_abs_diff_vs_torchmetrics_cpu"] = check(net, path, args.check_pair)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fixer runner for `vauss fill`: repairs splat renders with a single-step diffusion fixer, and scores how much each
repaired pixel can be trusted. `vauss fill` writes this file next to its work dir and runs it with the Python you
name (`--fixer-python`); it never imports anything from the splat repo.

  python splat_fixer.py fix --model fixer|difix_ref --in RENDERS/ --out FIXED/ [--ref REFS/] [--ckpt stock|PATH] [--shard I/N]
  python splat_fixer.py provenance --model fixer|difix_ref [--ckpt stock|PATH] --out fixer.json
  python splat_fixer.py confidence --a FITTED/ --b RENDERS/ --out CONF/ [--lp-tau 0.8] [--lp-floor 0.3] [--shard I/N]

--shard I/N takes every N-th view starting at I (one process per GPU, or per machine on a shared dir). Views whose output
already exists are skipped, so a run can be restarted.

The fixer protocol (what any other fixer has to do to plug in, see `vauss fill --fixer-cmd`):
  * input: a directory of PNG renders (RGB, any size), optionally a directory of reference photos with the same names;
  * output: a directory with one repaired PNG per input, same name, same size.

Models:
  fixer      NVIDIA Fixer (Cosmos-Predict2 0.6B, one step; github.com/nv-tlabs/Fixer, weights huggingface.co/nvidia/Fixer).
             Code Apache-2.0; weights under the NVIDIA Open Model License (commercial use allowed; "Built on NVIDIA
             Cosmos" attribution required). Default of `vauss fill`. Needs FIXER_SRC (the repo's src/) and FIXER_MODELS
             (the downloaded weights); setup_fixer.sh installs both and records their revision in FIXER_MODELS/MANIFEST.json.
             --ckpt stock = the pretrained weights, or a fine-tuned {state_dict_unet, state_dict_vae} checkpoint.
  difix_ref  Difix3D+ with a reference view (nvidia/difix_ref). NON-COMMERCIAL licence: research use only. Needs
             DIFIX_SRC (github.com/nv-tlabs/Difix3D src/).

confidence: per pixel, 1 - LPIPS-vgg(a, b) / lp_tau clamped to [lp_floor, 1], written as an 8-bit grey PNG. A fixer that
rewrites a well-covered area a lot is likely inventing something the other views won't agree with.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image


def log(m):
    print(f'{time.strftime("%H:%M:%S")} [fixer] {m}', flush=True)


def names_in(d, shard='0/1'):
    i, n = (int(v) for v in shard.split('/'))
    return sorted(n_ for n_ in os.listdir(d) if n_.lower().endswith('.png'))[i::n]


class NvidiaFixer:
    """NVIDIA Fixer at 1024 px wide (height keeps the aspect, rounded to 16), bf16, model kept loaded."""

    def __init__(self, ckpt='stock', res=1024, timestep=250):
        import torch
        sys.path.insert(0, os.environ['FIXER_SRC'])
        models = os.environ['FIXER_MODELS']
        from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo
        self.torch, self.res = torch, res
        path = models + '/pretrained/pretrained_fixer.pkl' if ckpt in (None, '', 'stock') else ckpt
        m = Pix2Pix_Turbo(pretrained_path=None, timestep=timestep, vae_skip_connection=False, batch_size=1)
        sd = torch.load(path, map_location='cpu', weights_only=False)
        for name, mod in (('unet', m.unet), ('vae', m.vae)):
            r = mod.load_state_dict(sd[f'state_dict_{name}'], strict=False)
            unexp = [k for k in r.unexpected_keys if not k.endswith('_extra_state')]
            miss = [k for k in r.missing_keys if not k.endswith('_extra_state') and 'dit_ema' not in k]
            assert not unexp and not miss, f'checkpoint does not match the model: {unexp[:3]} {miss[:3]}'
        del sd
        m.set_eval()
        self.m = m.to('cuda', dtype=torch.bfloat16)

    def __call__(self, img, ref=None):
        torch = self.torch
        w, h = img.size
        W, H = self.res, max(16, int(round(self.res * h / w / 16)) * 16)
        a = np.asarray(img.convert('RGB').resize((W, H), Image.BICUBIC), dtype=np.float32) / 127.5 - 1
        x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            y = self.m(x.to(torch.bfloat16)).float().clamp(-1, 1)
        out = ((y[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(out).resize((w, h), Image.BICUBIC)


class Difix:
    """Difix3D+ (NON-COMMERCIAL weights), one step at timestep 199, with the reference view for difix_ref."""

    def __init__(self, model='difix_ref'):
        import torch
        sys.path.insert(0, os.environ['DIFIX_SRC'])
        from pipeline_difix import DifixPipeline
        self.torch, self.model = torch, model
        self.pipe = DifixPipeline.from_pretrained(f'nvidia/{model}', trust_remote_code=True).to('cuda')
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(self, img, ref=None):
        kw = dict(ref_image=ref) if self.model == 'difix_ref' and ref is not None else {}
        with self.torch.no_grad():
            return self.pipe('remove degradation', image=img, num_inference_steps=1, timesteps=[199], guidance_scale=0.0,
                             height=img.height, width=img.width, **kw).images[0]


def provenance(a):
    """what repaired the views: model, checkpoint identity, weights revision and licence"""
    import hashlib, json
    rec = dict(model=a.model)
    if a.model == 'fixer':
        models = os.environ['FIXER_MODELS']
        path = models + '/pretrained/pretrained_fixer.pkl' if a.ckpt in (None, '', 'stock') else a.ckpt
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        rec.update(ckpt=a.ckpt, ckpt_path=path, ckpt_sha256=h.hexdigest(), weights='nvidia/Fixer',
                   licence='NVIDIA Open Model License (Built on NVIDIA Cosmos)')
        try:
            rec['manifest'] = json.load(open(models + '/MANIFEST.json'))
        except OSError:
            pass
    else:
        rec.update(weights=f'nvidia/{a.model}', licence='NVIDIA non-commercial (Difix3D+ model card): research use only')
    return rec


def cmd_provenance(a):
    """write provenance(a) to a.out once (vauss fill calls this before starting the fixer workers)"""
    import json
    tmp = f'{a.out}.{os.getpid()}.tmp'
    json.dump(provenance(a), open(tmp, 'w'), indent=1)
    os.replace(tmp, a.out)


def cmd_fix(a):
    names = names_in(a.inp, a.shard)
    os.makedirs(a.out, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(a.out, n))]
    log(f'{a.model}: {len(names)} renders, {len(todo)} to fix')
    if not todo:
        return
    t0 = time.time()
    fx = NvidiaFixer(a.ckpt) if a.model == 'fixer' else Difix(a.model)
    log(f'loaded {a.model} in {time.time() - t0:.0f} s')
    t0 = time.time()
    for i, n in enumerate(todo):
        img = Image.open(os.path.join(a.inp, n)).convert('RGB')
        ref = None
        if a.ref and os.path.exists(os.path.join(a.ref, n)):
            ref = Image.open(os.path.join(a.ref, n)).convert('RGB')
        out = fx(img, ref)
        if out.size != img.size:
            out = out.resize(img.size, Image.LANCZOS)
        tmp = os.path.join(a.out, '.' + n)
        out.save(tmp, format='PNG')
        os.replace(tmp, os.path.join(a.out, n))
        if i % 100 == 0:
            log(f'fixed {i + 1}/{len(todo)}')
    log(f'fixed {len(todo)} in {time.time() - t0:.0f} s ({1000 * (time.time() - t0) / len(todo):.0f} ms/view)')


def cmd_confidence(a):
    import torch
    import torch.nn.functional as F
    import lpips
    net = lpips.LPIPS(net='vgg', spatial=True, verbose=False).cuda().eval()
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in names_in(a.a, a.shard) if not os.path.exists(os.path.join(a.out, n))]
    t0 = time.time()
    for n in names:
        def load(d):
            x = np.asarray(Image.open(os.path.join(d, n)).convert('RGB'), dtype=np.float32) / 127.5 - 1
            return torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
        x, y = load(a.a), load(a.b)
        with torch.no_grad():
            m = net(x, y)
            m = F.interpolate(m, size=x.shape[2:], mode='bilinear', align_corners=False)[0, 0]
        conf = (1 - m / a.lp_tau).clamp(a.lp_floor, 1)
        tmp = os.path.join(a.out, '.' + n)
        Image.fromarray((conf * 255).round().byte().cpu().numpy()).save(tmp, format='PNG')
        os.replace(tmp, os.path.join(a.out, n))
    log(f'confidence for {len(names)} views in {time.time() - t0:.0f} s')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fix')
    f.add_argument('--model', default='fixer', choices=['fixer', 'difix_ref', 'difix'])
    f.add_argument('--ckpt', default='stock')
    f.add_argument('--in', dest='inp', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--ref', default=None)
    f.add_argument('--shard', default='0/1')
    p = sub.add_parser('provenance')
    p.add_argument('--model', default='fixer', choices=['fixer', 'difix_ref', 'difix'])
    p.add_argument('--ckpt', default='stock')
    p.add_argument('--out', required=True)
    c = sub.add_parser('confidence')
    c.add_argument('--a', required=True)
    c.add_argument('--b', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--lp-tau', type=float, default=0.8)
    c.add_argument('--lp-floor', type=float, default=0.3)
    c.add_argument('--shard', default='0/1')
    a = ap.parse_args()
    {'fix': cmd_fix, 'provenance': cmd_provenance, 'confidence': cmd_confidence}[a.cmd](a)


if __name__ == '__main__':
    main()

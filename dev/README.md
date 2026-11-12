# Developer tools

These helpers support testing and release validation; they are not required to install or run splat.

- [`free-roam/`](free-roam/README.md): standalone viewer packaging, collision/navigation checks,
  and browser recordings. Requires Python with NumPy; navigation checks need Node and Three.js.
- `proof/`: Windows release-archive validation on a software GPU, synthetic scene generation,
  and a small standalone wgpu adapter probe. See the comments in `proof/windows-pipeline.ps1`.
- `patch-viewer-roam.py`: updates a copied legacy viewer package. Usage and precautions are in
  [`scripts/patch-viewer-roam.md`](../scripts/patch-viewer-roam.md).

Run the portable viewer checks from the repository root:

```sh
python3 -m unittest discover -s dev/free-roam -p 'test_*.py' -v
```

Set `THREE_MODULE=/path/to/three.module.js` to include the navigation test when Three.js is not
installed as a local npm package. Browser recording additionally requires Playwright and Chromium.

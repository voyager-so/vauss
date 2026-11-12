# Contributing to Vauss

Thanks for helping. Bug reports, fixes and small focused features are all welcome.

## Before you start

- For anything bigger than a bug fix, open an issue first so we can agree on the approach.
- Security problems: see [SECURITY.md](SECURITY.md). Don't open a public issue.

## Building and checking

```sh
cargo build --locked --workspace
cargo test --locked -p <crate you changed>
cargo deny check licenses bans sources                                                   # licence policy (deny.toml)
```

Training needs a GPU (wgpu: Vulkan, Metal or DX12). Tests that need real data or a GPU are
skipped without them.

## Pull requests

This repository is the public source mirror. Open issues and pull requests here; you do not need
access to another repository. Maintainers review contributions here, integrate accepted changes into
the upstream development repository, and publish a new public snapshot. A contribution PR may be
closed with a link to that snapshot instead of merging its commits directly into this mirror.

- One topic per PR. Explain what changed and why, and how you checked it: a test, or the
  command you ran and what it printed. A change that affects quality or speed needs a measured
  before/after on the same data.
- Keep the surrounding style. `cargo fmt` the files you touched.
- Don't commit data you don't have the right to share: no private footage, and no datasets whose
  licence forbids redistribution. Public example clips and their licences are listed in
  [examples/DATA.md](examples/DATA.md).
- New dependencies must pass `cargo deny check licenses bans sources`. No GPL, AGPL or LGPL code
  linked into Vauss, and no non-commercial model weights as defaults.

## Licence

Original Vauss code is source-available under the [Vauss Preview License](LICENSE), not open source.
Read the Contributions section of the [license](LICENSE) before submitting changes. Separately licensed
components retain their own terms: Brush-derived code remains Apache-2.0, and vendored CubeCL
remains MIT OR Apache-2.0. Preserve component license files, attribution headers and [NOTICE](NOTICE);
the Preview License does not replace those upstream grants.

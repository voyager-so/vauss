# Contributing to splat

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

- One topic per PR. Explain what changed and why, and how you checked it: a test, or the
  command you ran and what it printed. A change that affects quality or speed needs a measured
  before/after on the same data.
- Keep the surrounding style. `cargo fmt` the files you touched.
- Don't commit data you don't have the right to share: no private footage, and no datasets whose
  licence forbids redistribution. Public example clips and their licences are listed in
  [examples/DATA.md](examples/DATA.md).
- New dependencies must pass `cargo deny check licenses bans sources`. No GPL, AGPL or LGPL code
  linked into splat, and no non-commercial model weights as defaults.

## Licence

splat is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). By contributing you agree that
your contribution is licensed under the same terms.

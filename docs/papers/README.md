# Benchmark evidence

[Historical benchmark and parity summary](parity.md) records measured comparisons, including
negative and provisional outcomes. The linked files in `tools/verified/` contain the numerical
verification records and their cited dependencies. These historical studies are distinct from
the current launch measurements in the repository README.

## Verification

`tools/verify.py` and `tools/verify_pose.py` check supplied per-view measurements. Start with
`python3 docs/papers/tools/verify.py --help`. `tools/eq_2fold.py` supports cross-validated
re-scoring when the original per-view inputs are available; set `SPLAT_DATA` to your own data root.
The raw experiment corpus and original private builds are not included, so the retained summary
records alone do not reproduce historical training runs. Content checksums remain in the records;
private commit identifiers are replaced consistently by descriptive historical-build labels.

[Protocols](claims/protocols/) document evaluation design and caveats. A result requires identical
view coverage, a declared metric and scorer, explicit seeds, hardware and budget, and the stated
comparison rule. Comparing methods at equal steps, equal splat caps and equal image quality answers
different questions; keep those comparisons separate.

## Data and attribution

Download benchmark data from its original authors and follow each dataset's terms. This repository
does not redistribute benchmark captures or restricted model weights. Dataset permission does not
follow automatically from the Apache-2.0 code license. Research-only methods and datasets mentioned
in comparisons are not bundled product dependencies.

Unpublished drafts, the internal planning ledger and private experiment orchestration are omitted.

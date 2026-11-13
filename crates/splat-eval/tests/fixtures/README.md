# Geometry evidence fixtures

These are reductions of the original COLMAP binary models, not invented camera paths.
`geometry_evidence` uses every observed 3D point to compute each camera's median positive depth,
invalid-depth count and observation-component connectivity. The checked-in JSON retains camera
centres, frame names and those statistics (no source photographs or splat payloads).

Regenerate with `cargo run --locked -p splat-eval --example geometry_evidence -- MODEL_DIR`.
The full binary models were also evaluated using `splat-eval sfm`, independently of fixture tests:
failed179 rejects (depth p90/p10 95340.26634), good241 passes all health gates (1.453405028).

| Model | Binary | SHA-256 |
|---|---|---|
| failed179 | cameras.bin | ef411c63fd71fa606996e702b62ecb5bf10d79a9c5ba8a9942bdd5d0142e501a |
| failed179 | images.bin | bafab9a3b1e1b0abb6b1247bf4935aa5cf14e86a7e6e8a9ec873f65904dc1684 |
| failed179 | points3D.bin | fe21c741b52883bf8ffc37f2d0d469ff417a54a5a7c5377d20042507249835a4 |
| good241 | cameras.bin | 3689944eaba96978f152e8107873e4eee6434381ad676700a382092743b91cb2 |
| good241 | images.bin | 87267507573cb06541346cd5bcf806d09320ae7e186a0f5c9182195141816c82 |
| good241 | points3D.bin | 78df7aec5cc594ac8d814f077276704cf36d037943e1b1198b1f0733cbf67e35 |

Source: Vauss demo fix evidence, `couple-diagnosis/models.tar.gz:sparse/0` and
`fal-shared-capture/final-model/cameras/0`, respectively. Synthetic full-model tests independently
exercise observation lookup, disconnected components, negative depths, translation impulses and
stationary cameras, so the fixture assertions do not merely restate cached pass/fail decisions.

# Collision generation dependencies

- meshopt Rust bindings **0.6.2**, MIT OR Apache-2.0; bundled meshoptimizer, MIT.
  https://github.com/gwihlidal/meshopt-rs — https://github.com/zeux/meshoptimizer
- Parry **0.25.3** (`parry3d-f64`), Apache-2.0: pure Rust V-HACD implementation and convex hulls.
  https://github.com/dimforge/parry — https://docs.rs/parry3d-f64/0.25.3/parry3d_f64/transformation/vhacd/
  This uses Parry's implementation, not the original C++ V-HACD library (BSD-3-Clause).
- UnityGaussianSplatting, MIT, is referenced only and not redistributed.
  https://github.com/aras-p/UnityGaussianSplatting

The generated mesh/PLY files contain scene data; these dependency licenses do not replace
or grant rights to the source scene. Check the source scene's license before redistribution.

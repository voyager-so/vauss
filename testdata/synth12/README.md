# synth12

Twelve 640x480 views of a small synthetic scene (a textured room holding three textured
boxes, cameras on a 100-degree arc), used by the Windows release proof job in
`.github/workflows/release.yml` to run the real pipeline (ingest, COLMAP SfM, Brush
training, evaluation) on a GPU-less runner.

Provenance: rendered by `dev/proof/make_synthetic_scene.py` (numpy + Pillow, fixed
seeds); regenerate with

    python3 dev/proof/make_synthetic_scene.py testdata/synth12/images --width 640 --height 480

No third-party content: the images are original to this repository and carry its license.
We did not commit Tanks and Temples frames because that dataset's license page both grants
CC BY 4.0 and forbids making the data available to third parties without permission.

Checked locally with COLMAP 4.2.1 (CPU SIFT, exhaustive matching, PINHOLE single camera):
12/12 images registered, 849 points, mean reprojection error 0.22 px.

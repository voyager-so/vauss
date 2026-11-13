Retained calibration from a 25-view authored-camera capture. Original metadata
SHA-256:
`e75ccc6273e9791ea77a856931f10f3a43adc8f8ef6ff919f4fdc9f0547ee455`.

The regression generates placeholder pixels at the original dimensions; it makes
no image-quality or training-quality claim. All 25 authored transforms, unequal
fx/fy, and explicit test_filenames are unchanged. Four non-collinear world
landmarks are added as test probes in front of the first camera. The CPU test
checks extraction, skip-SfM dispatch with unavailable helpers, frozen membership,
the actual worker dataset loader, and the web exporter. No GPU training is run.
Tolerances: source/COLMAP/export rigid poses 2e-6; float worker poses 3e-6;
worker K 2e-4 pixels; export K 1e-9 pixels; landmark projection 0.002 pixels.

`solved-cameras.json` retains the exact failed camera from a default SfM run,
including the source camera binary SHA-256. At 1024×576 it solved
fx=1348.5809478553736 and fy=8406.5200824053, exhibiting more than 6× focal-aspect
drift relative to the supplied calibration.

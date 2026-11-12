@echo off
rem COLMAP shim for GPU-less CI runners: forwards every call to %COLMAP_REAL% and adds
rem use_gpu 0 to SIFT extraction and matching. Used only when the product's default COLMAP
rem invocation fails on the runner (see dev/proof/windows-pipeline.ps1).
setlocal
set "EXTRA="
if /I "%~1"=="feature_extractor" set "EXTRA=--FeatureExtraction.use_gpu 0"
if /I "%~1"=="exhaustive_matcher" set "EXTRA=--FeatureMatching.use_gpu 0"
if /I "%~1"=="sequential_matcher" set "EXTRA=--FeatureMatching.use_gpu 0"
echo [colmap-cpu] "%COLMAP_REAL%" %* %EXTRA% 1>&2
"%COLMAP_REAL%" %* %EXTRA%
exit /b %ERRORLEVEL%

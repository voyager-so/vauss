<#
Windows end-to-end proof for a release archive on a clean, GPU-less runner.

Stages: verify + unpack the archive, offline `splat doctor` (bundled tools/ must make it
ready), then `splat run --driver fixed` on a tiny image set. Training runs on whatever wgpu
adapter the product picks by default (on a GitHub runner: DX12 WARP, see -Probe).

When an attempt fails, the failing tool decides the next attempt, each fix applied once and
recorded in the summary:
  reconstruct_sfm -> SPLAT_COLMAP = colmap-cpu.cmd (SIFT extraction/matching with use_gpu 0)
  train_splats / evaluate -> the Windows SDK's dxcompiler.dll on PATH (wgpu DX12 uses DXC
                             from PATH, else falls back to FXC)
Every attempt's run dir (minus files over 25 MB), logs, doctor JSON and a summary land in -Out.
Exits 1 unless some attempt finished the pipeline.
#>
param(
    [Parameter(Mandatory)] [string]$ArchiveDir,
    [Parameter(Mandatory)] [string]$Images,
    [Parameter(Mandatory)] [string]$Out,
    [string]$Probe,
    [string]$Work = (Join-Path ($env:RUNNER_TEMP ?? [IO.Path]::GetTempPath()) 'splat-proof'),
    [int]$Steps = 200,
    [string]$TimeBudget = '40m'
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $Out, $Work | Out-Null
$Out = (Resolve-Path $Out).Path
$Work = (Resolve-Path $Work).Path
$Images = (Resolve-Path $Images).Path
$stages = [System.Collections.Generic.List[object]]::new()
function Add-Stage([string]$Name, [bool]$Ok, [string]$Evidence) {
    $stages.Add([ordered]@{ stage = $Name; ok = $Ok; evidence = $Evidence })
    Write-Host "== $Name : $(if ($Ok) { 'PASS' } else { 'FAIL' }) :: $Evidence"
}
function Write-Summary {
    $summary = [ordered]@{ stages = $stages; attempts = $script:attempts }
    $summary | ConvertTo-Json -Depth 8 | Out-File -Encoding utf8NoBOM (Join-Path $Out 'summary.json')
    $md = @('| Stage | Result | Evidence |', '| --- | --- | --- |')
    foreach ($s in $stages) {
        $text = ($s.evidence -replace '\|', '/' -replace "`r?`n", ' ')
        $md += "| $($s.stage) | $(if ($s.ok) { 'pass' } else { 'FAIL' }) | $text |"
    }
    $md -join "`n" | Out-File -Encoding utf8NoBOM (Join-Path $Out 'summary.md')
    if ($env:GITHUB_STEP_SUMMARY) { "## Windows pipeline proof`n" + ($md -join "`n") | Out-File -Append -Encoding utf8NoBOM $env:GITHUB_STEP_SUMMARY }
}
$script:attempts = @()

# --- 1. Verify and unpack the archive -------------------------------------------------
$zip = Get-ChildItem $ArchiveDir -Recurse -Filter 'splat-*.zip' | Select-Object -First 1
if (-not $zip) { Add-Stage 'archive unpack' $false "no splat-*.zip under $ArchiveDir"; Write-Summary; exit 1 }
$expected = (Get-Content "$($zip.FullName).sha256" -Raw).Split(' ')[0].Trim()
$actual = (Get-FileHash -Algorithm SHA256 $zip.FullName).Hash.ToLowerInvariant()
$pkg = Join-Path $Work 'pkg'
Remove-Item $pkg -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $zip.FullName $pkg
$splat = (Get-ChildItem $pkg -Recurse -Filter splat.exe | Select-Object -First 1).FullName
if (-not $splat) { Add-Stage 'archive unpack' $false "no splat.exe in $($zip.Name)"; Write-Summary; exit 1 }
Get-ChildItem $pkg -Recurse -File | ForEach-Object { '{0,12}  {1}' -f $_.Length, $_.FullName.Substring($pkg.Length + 1) } |
    Out-File -Encoding utf8NoBOM (Join-Path $Out 'archive-files.txt')
$version = (& $splat --version) -join ' '
Add-Stage 'archive unpack' (($expected -eq $actual) -and $LASTEXITCODE -eq 0) "$($zip.Name) sha256 $(if ($expected -eq $actual) { 'matches' } else { "MISMATCH $actual != $expected" }); $version; file list archive-files.txt"

# --- 2. Offline doctor: empty cache, no token ------------------------------------------
$env:SPLAT_CACHE_DIR = Join-Path $Work 'cache'
Remove-Item $env:SPLAT_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item Env:GITHUB_TOKEN, Env:SPLAT_GITHUB_TOKEN -ErrorAction SilentlyContinue
$doctorPath = Join-Path $Out 'doctor.json'
& $splat doctor | Out-File -Encoding utf8NoBOM $doctorPath
$doctorExit = $LASTEXITCODE
$r = (Get-Content $doctorPath -Raw | ConvertFrom-Json).readiness
$sources = [ordered]@{
    ffmpeg        = $r.required.ffmpeg.source
    colmap        = "$($r.required.colmap.status)$(if ($r.required.colmap.source) { '/' + $r.required.colmap.source })"
    onnxruntime   = $r.optional.onnxruntime.source
    'lpips-alex'  = $r.optional.'lpips-alex'.source
    'lpips-vgg'   = $r.optional.'lpips-vgg'.source
}
$bundledOk = $doctorExit -eq 0 -and $r.ready -eq $true -and
    (@('ffmpeg', 'onnxruntime', 'lpips-alex', 'lpips-vgg') | Where-Object { $sources[$_] -ne 'bundled' }).Count -eq 0
Add-Stage 'doctor ready' $bundledOk ("ready=$($r.ready); " + (($sources.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ', ') + '; doctor.json')

# --- 3. wgpu adapter the product would pick (probe built from the same wgpu rev) ---------
$adapter = 'unknown (no probe)'
if ($Probe -and (Test-Path $Probe)) {
    $probeOut = Join-Path $Out 'wgpu-probe.json'
    & $Probe 2> (Join-Path $Out 'wgpu-probe.stderr.txt') | Out-File -Encoding utf8NoBOM $probeOut
    $probeExit = $LASTEXITCODE
    try {
        $p = Get-Content $probeOut -Raw | ConvertFrom-Json
        $adapter = "$($p.auto_selected.name) ($($p.auto_selected.backend), $($p.auto_selected.device_type), subgroups=$($p.auto_selected.subgroups))"
        Add-Stage 'wgpu adapter probe' ($probeExit -eq 0) "settled_api=$($p.settled_api); selected=$adapter; request_device=$($p.request_device); wgpu-probe.json"
    } catch {
        Add-Stage 'wgpu adapter probe' $false "probe exit $probeExit, unparsable output; wgpu-probe.stderr.txt"
    }
}

# --- 4. Pipeline attempts ---------------------------------------------------------------
function Get-StopReason([string]$RunDir) {
    $journal = Join-Path $RunDir 'journal.jsonl'
    if (-not (Test-Path $journal)) { return 'no journal' }
    $finished = Get-Content $journal | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.event -eq 'run_finished' -or $_.type -eq 'run_finished' -or $_.kind -eq 'run_finished' } | Select-Object -Last 1
    if (-not $finished) { return (Get-Content $journal -Tail 3) -join ' ' }
    $payload = $finished.data ?? $finished.payload ?? $finished
    return "$($payload.status): $($payload.stop_reason)"
}
function Find-Dxc {
    $kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    Get-ChildItem $kits -Recurse -Filter dxcompiler.dll -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -like '*\x64' } | Sort-Object FullName -Descending | Select-Object -First 1
}
$fixes = [ordered]@{}
$success = $false
for ($i = 1; $i -le 3 -and -not $success; $i++) {
    $name = if ($fixes.Count -eq 0) { 'default' } else { ($fixes.Keys -join '+') }
    $runDir = Join-Path $Work "runs\$i-$name"
    $logDir = Join-Path $Out "attempt-$i-$name"
    New-Item -ItemType Directory -Force $logDir | Out-Null
    Write-Host "== attempt $i ($name): $runDir"
    $started = Get-Date
    & $splat run $Images --goal 'Windows CI proof: tiny synthetic scene' --driver fixed --steps $Steps `
        --target-count 20 --max-edge 640 --no-mesh --time-budget $TimeBudget --run-dir $runDir `
        > (Join-Path $logDir 'run.stdout.json') 2> (Join-Path $logDir 'run.stderr.txt')
    $exit = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds)
    if (Test-Path $runDir) {
        & $splat logs --run-dir $runDir *> (Join-Path $logDir 'splat-logs.txt')
        & $splat status --run-dir $runDir *> (Join-Path $logDir 'splat-status.json')
        # Run dir minus anything huge; robocopy exit codes below 8 are success.
        robocopy $runDir (Join-Path $logDir 'run') /E /MAX:25000000 /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { Write-Warning "robocopy failed ($LASTEXITCODE)" }
    }
    $reason = if ($exit -eq 0) { 'succeeded' } else { Get-StopReason $runDir }
    $script:attempts += [ordered]@{ attempt = $i; fixes = @($fixes.Keys); fix_detail = @($fixes.Values); exit = $exit; seconds = $elapsed; result = $reason }
    Write-Host "== attempt $i exit=$exit after ${elapsed}s: $reason"
    if ($exit -eq 0) { $success = $true; break }
    # The failing tool: the last operation that did not succeed.
    $state = Join-Path $runDir 'state.json'
    $failing = if (Test-Path $state) {
        @((Get-Content $state -Raw | ConvertFrom-Json).operations.PSObject.Properties.Value) |
            Where-Object { $_.result.status -ne 'succeeded' } | Select-Object -Last 1 | ForEach-Object { $_.tool }
    }
    $script:attempts[-1]['failing_tool'] = $failing
    if ($failing -eq 'reconstruct_sfm' -and -not $fixes.Contains('cpu-sift')) {
        $colmap = (& $splat --json fetch colmap | ConvertFrom-Json).path
        # Diagnose outside the product: the same COLMAP on a plain (non \\?\) path, GPU SIFT
        # default vs use_gpu 0, so the failure is pinned to the GPU default, not the path form.
        $plain = $colmap -replace '^\\\\\?\\', ''
        foreach ($probe in @(@{ name = 'gpu-default'; extra = @() }, @{ name = 'use_gpu-0'; extra = @('--FeatureExtraction.use_gpu', '0') })) {
            $db = Join-Path $Work "colmap-$($probe.name).db"
            Remove-Item $db -ErrorAction SilentlyContinue
            $log = Join-Path $Out "colmap-feature_extractor-$($probe.name).log"
            & $plain feature_extractor --database_path $db --image_path $Images --ImageReader.camera_model PINHOLE @($probe.extra) *> $log
            $code = $LASTEXITCODE
            $tail = ((Get-Content $log -Tail 3) -join ' ').Trim()
            Add-Stage "COLMAP feature_extractor direct ($($probe.name))" ($code -eq 0) ('exit 0x{0:X8}; {1}; {2}' -f $code, $tail, (Split-Path $log -Leaf))
        }
        $env:COLMAP_REAL = $plain
        $env:SPLAT_COLMAP = Join-Path $PSScriptRoot 'colmap-cpu.cmd'
        $fixes['cpu-sift'] = "SPLAT_COLMAP=colmap-cpu.cmd wrapping $plain with --FeatureExtraction.use_gpu 0 / --FeatureMatching.use_gpu 0"
    } elseif ($failing -in 'train_splats', 'evaluate' -and -not $fixes.Contains('dxc')) {
        $dxc = Find-Dxc
        if (-not $dxc) { Write-Host '== no dxcompiler.dll in Windows Kits; no further fix'; break }
        $env:PATH = "$($dxc.DirectoryName);$env:PATH"
        $fixes['dxc'] = "PATH += $($dxc.DirectoryName) (dxcompiler.dll)"
    } else {
        break
    }
}

# --- 5. Per-stage results from the last attempt's run ------------------------------------
$last = $script:attempts[-1]
$lastDir = Join-Path $Out "attempt-$($last.attempt)-$(if ($last.fixes.Count) { $last.fixes -join '+' } else { 'default' })"
$statePath = Join-Path $lastDir 'run/state.json'
$ops = if (Test-Path $statePath) { @((Get-Content $statePath -Raw | ConvertFrom-Json).operations.PSObject.Properties.Value) } else { @() }
foreach ($tool in 'extract_frames', 'reconstruct_sfm', 'train_splats', 'evaluate') {
    $op = $ops | Where-Object { $_.tool -eq $tool } | Select-Object -Last 1
    $label = @{ extract_frames = 'ingest (extract_frames)'; reconstruct_sfm = 'SfM (COLMAP)'; train_splats = "train ($Steps steps; adapter $adapter)"; evaluate = 'eval' }[$tool]
    if (-not $op) { Add-Stage $label $false 'not reached'; continue }
    $status = $op.result.status
    $detail = if ($status -eq 'succeeded') { $op.result.metrics | ConvertTo-Json -Compress -Depth 4 } else { "$($op.result.error.code): $($op.result.error.message)" }
    if ($detail.Length -gt 700) { $detail = $detail.Substring(0, 700) + '...' }
    Add-Stage $label ($status -eq 'succeeded') "$status; $detail; state.json in $(Split-Path $lastDir -Leaf)"
}
foreach ($a in $script:attempts) {
    Add-Stage "attempt $($a.attempt) [$(if ($a.fixes.Count) { $a.fixes -join '+' } else { 'default' })]" ($a.exit -eq 0) "exit $($a.exit) after $($a.seconds)s: $($a.result)"
}
$report = Join-Path $lastDir 'run.stdout.json'
if ($success -and (Test-Path $report)) {
    $rep = Get-Content $report -Raw | ConvertFrom-Json
    $eval = @($rep.evaluations)[-1]
    if ($eval) {
        Add-Stage 'eval metrics (report.json)' $true (($eval.metrics | ConvertTo-Json -Compress -Depth 4) -replace '(.{600}).+', '$1...')
        $renders = Join-Path $Out 'renders'
        New-Item -ItemType Directory -Force $renders | Out-Null
        foreach ($img in @($eval.images)) {
            $src = Join-Path $lastDir "run/$img"
            if (Test-Path $src) { Copy-Item $src (Join-Path $renders ((Split-Path (Split-Path $src) -Leaf) + '-' + (Split-Path $src -Leaf))) }
        }
    }
}
Write-Summary
exit $(if ($success) { 0 } else { 1 })

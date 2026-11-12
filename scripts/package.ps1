param([string]$Target = 'x86_64-pc-windows-msvc', [string]$Output = 'target/dist')
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if ($Target -ne 'x86_64-pc-windows-msvc') { throw "Unsupported Windows target: $Target" }
New-Item -ItemType Directory -Force $Output | Out-Null
$Output = (Resolve-Path $Output).Path
$metadata = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
$version = ($metadata.packages | Where-Object name -eq 'splat-contracts').version
$name = "vauss-v$version-$Target"
cargo build --locked --release -p splat-cli --bin vauss --target $Target
if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
$licenses = Join-Path $Output 'THIRD_PARTY_LICENSES.txt'
cargo about generate --locked --fail --manifest-path crates/splat-cli/Cargo.toml --config dist/about.toml dist/about.hbs --output-file $licenses
if ($LASTEXITCODE -ne 0) { throw 'cargo-about failed; no archive created' }
$stage = Join-Path $Output ".stage-$([Guid]::NewGuid().ToString('N'))"
try {
    $root = Join-Path $stage $name
    New-Item -ItemType Directory -Force $root | Out-Null
    # Git Bash, not WSL bash: the staging script is shared with the Unix packager.
    & (Join-Path $env:ProgramFiles 'Git/bin/bash.exe') scripts/bundle-tools.sh $Target ((Join-Path $root 'tools') -replace '\\', '/')
    if ($LASTEXITCODE -ne 0) { throw 'bundling runtime tools failed' }
    Copy-Item "target/$Target/release/vauss.exe" $root
    Copy-Item "$root/vauss.exe" "$root/splat.exe"
    Copy-Item "$root/vauss.exe" "$root/voyager-splat.exe"
    Copy-Item $licenses $root
    Copy-Item 'dist/LICENSE.txt' $root
    # Apache-2.0 4(d): ship the top-level NOTICE plus every NOTICE it points to.
    $notices = 'NOTICE', 'crates/splat-train/NOTICE', 'crates/splat-eval/NOTICE', 'crates/splat-sfm-learned/NOTICE', 'crates/splat-sfm-learned/LICENSE-IJG.txt', 'vendor/cubecl-common/NOTICE.md', 'vendor/cubecl-wgpu/NOTICE.md'
    ($notices | ForEach-Object { "==> $_ <==`n" + (Get-Content -Raw $_) }) -join "`n" |
        Set-Content -Encoding utf8NoBOM (Join-Path $root 'NOTICE.txt')
    Copy-Item 'docs/distribution.md' $root
    Copy-Item 'crates/splat-fetch/assets.json' $root
    Copy-Item 'dist/README-windows.txt' (Join-Path $root 'README.txt')
    $archive = Join-Path $Output "$name.zip"
    Compress-Archive -Path $root -DestinationPath $archive -Force
    $hash = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText("$archive.sha256", "$hash  $name.zip`n")
    Write-Output $archive
} finally { Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue }

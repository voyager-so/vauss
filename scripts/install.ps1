param([Parameter(Mandatory=$true)][string]$Version,
      [string]$InstallDir = "$env:LOCALAPPDATA\vauss\bin")
$ErrorActionPreference = 'Stop'
$Version = $Version.TrimStart('v')
if ($Version -notmatch '^[0-9A-Za-z.+-]+$') { throw 'Invalid version' }
if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') { throw 'This installer requires Windows x64' }
$name = "vauss-v$Version-x86_64-pc-windows-msvc"
$base = if ($env:VAUSS_RELEASE_BASE_URL) { $env:VAUSS_RELEASE_BASE_URL.TrimEnd('/') } elseif ($env:SPLAT_RELEASE_BASE_URL) { $env:SPLAT_RELEASE_BASE_URL.TrimEnd('/') } else { 'https://github.com/voyager-so/vauss/releases/download' }
if (-not $base.StartsWith('https://')) { throw 'Release URL must use HTTPS' }
$stage = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $stage | Out-Null
try {
    $archive = Join-Path $stage "$name.zip"
    Invoke-WebRequest "$base/vauss-v$Version/$name.zip" -OutFile $archive
    $checksum = (Invoke-WebRequest "$base/vauss-v$Version/$name.zip.sha256").Content
    $expected = ($checksum.Trim() -split '\s+')[0].ToLowerInvariant()
    if ($expected -notmatch '^[0-9a-f]{64}$') { throw 'Invalid checksum' }
    if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Checksum mismatch; nothing installed' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        $allowed = @("$name/", "$name/vauss.exe", "$name/splat.exe", "$name/voyager-splat.exe", "$name/LICENSE.txt", "$name/NOTICE.txt", "$name/THIRD_PARTY_LICENSES.txt", "$name/distribution.md", "$name/assets.json", "$name/README.txt")
        foreach ($entry in $zip.Entries) {
            $member = $entry.FullName.Replace('\','/')
            if ($member.Contains('..')) { throw "Unsafe archive member: $($entry.FullName)" }
            if ($member -notin $allowed -and -not $member.StartsWith("$name/tools/")) { throw "Unexpected archive member: $($entry.FullName)" }
        }
    } finally { $zip.Dispose() }
    Expand-Archive $archive $stage
    New-Item -ItemType Directory -Force $InstallDir | Out-Null
    $root = Join-Path $stage $name
    foreach ($command in @('vauss', 'splat', 'voyager-splat')) {
        Copy-Item "$root/vauss.exe" "$InstallDir/$command-new.exe" -Force
        Move-Item "$InstallDir/$command-new.exe" "$InstallDir/$command.exe" -Force
    }
    # Vauss resolves bundled helpers from tools\ beside vauss.exe.
    $tools = Join-Path $InstallDir 'tools'
    if (Test-Path $tools) { Remove-Item $tools -Recurse -Force }
    Copy-Item "$root/tools" $tools -Recurse
    $notices = Join-Path $InstallDir "../licenses/$Version"
    New-Item -ItemType Directory -Force $notices | Out-Null
    Copy-Item "$root/*.txt", "$root/distribution.md", "$root/assets.json" $notices
    Write-Host "Installed $InstallDir\vauss.exe. Add $InstallDir to PATH, then run: vauss doctor"
} finally { Remove-Item $stage -Recurse -Force }

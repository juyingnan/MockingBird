param(
    [string]$Manifest = "artifacts\manifests\ravdess.parquet",
    [string]$CacheRoot = "artifacts\embeddings",
    [int]$BatchSize = 4,
    [int]$SmokeSize = 16,
    [switch]$SmokeOnly,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Environment not found. Run scripts\setup.ps1 first."
}

$manifestPath = if ([System.IO.Path]::IsPathRooted($Manifest)) {
    $Manifest
} else {
    Join-Path $projectRoot $Manifest
}
$cachePath = if ([System.IO.Path]::IsPathRooted($CacheRoot)) {
    $CacheRoot
} else {
    Join-Path $projectRoot $CacheRoot
}

$arguments = @(
    "-m", "mockingbird2027", "extract", "wavlm", $manifestPath,
    "--cache-root", $cachePath,
    "--batch-size", $BatchSize,
    "--smoke-size", $SmokeSize,
    "--device", $Device
)
if ($SmokeOnly) {
    $arguments += "--smoke-only"
}

& $python @arguments
exit $LASTEXITCODE

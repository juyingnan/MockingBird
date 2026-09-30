param(
    [string]$Config = "configs\experiments\m4_wavlm_ravdess.json",
    [string]$OutputDir,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Environment not found. Run scripts\setup.ps1 first."
}
if (-not $ValidateOnly -and -not $OutputDir) {
    throw "Provide a new results\<run-id> output directory, or use -ValidateOnly."
}

Push-Location $projectRoot
try {
    $arguments = @("-u", "-m", "mockingbird2027", "intervene", "--config", $Config)
    if ($ValidateOnly) {
        $arguments += "--validate-only"
    } else {
        $arguments += @("--output-dir", $OutputDir)
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "M4 validation/run failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

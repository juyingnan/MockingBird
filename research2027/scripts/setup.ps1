param(
    [string]$BootstrapPython = "python"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$python = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $python)) {
    & $BootstrapPython -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

& $python --version
if ($LASTEXITCODE -ne 0) {
    throw "The project virtual environment does not contain a working Python interpreter."
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $python -m pip install --editable "$projectRoot[dev,wavlm]"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Environment ready: $venvPath"


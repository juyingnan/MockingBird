$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Environment not found. Run scripts\setup.ps1 first."
}

& $python -m pytest
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m mockingbird2027 --help
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}


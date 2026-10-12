# Build the installer on a Windows PC (the same steps as the CI workflow).
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#
# Needs Python 3.11 (the "py" launcher) and Inno Setup 6. Produces
# dist\PPEStation-Setup-<version>.exe.

param([switch]$SkipTests)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv-build)) { py -3.11 -m venv .venv-build }
$py = ".\.venv-build\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt -r packaging\requirements-build.txt -c packaging\constraints.txt

if (-not $SkipTests) {
    & $py -m pytest -q tests\test_paths.py tests\test_supervise.py tests\test_provision.py tests\test_config.py
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}

& $py -m PyInstaller --noconfirm --clean packaging\ppe_station.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Self-test the frozen program without touching this PC's real ProgramData.
$env:PPE_STATION_DATA = Join-Path $env:TEMP "ppe-selftest"
& dist\PPEStation\PPEStation-console.exe --selftest
$code = $LASTEXITCODE
Remove-Item Env:\PPE_STATION_DATA
if ($code -ne 0) { throw "self-test of the frozen build failed" }

$version = & $py -c "import ppe; print(ppe.__version__)"
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
& $iscc "/DAppVersion=$version" packaging\installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Write-Host "Built dist\PPEStation-Setup-$version.exe"

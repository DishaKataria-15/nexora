# Nexora setup for Windows PowerShell. Run from the project folder:   .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv }
elseif (Get-Command python -ErrorAction SilentlyContinue) { python -m venv .venv }
else { Write-Error "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'." }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Write-Host ""
Write-Host "Done. Start Nexora with:   .\scripts\run.ps1" -ForegroundColor Green

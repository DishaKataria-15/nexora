# Start Nexora (Windows PowerShell):   .\scripts\run.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path .\.venv\Scripts\python.exe)) { Write-Error "Run .\scripts\setup.ps1 first." }
& .\.venv\Scripts\python.exe -m streamlit run app.py

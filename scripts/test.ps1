# Run the automated tests (Windows PowerShell):   .\scripts\test.ps1
$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m pytest -q

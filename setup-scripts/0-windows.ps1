# Δημιουργεί .venv στη ρίζα του repo (Windows · PowerShell).
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if (Test-Path (Join-Path $RepoRoot ".git")) {
    Write-Host "→ git config core.hooksPath .githooks"
    git config core.hooksPath .githooks
} else {
    Write-Warning "Δεν βρέθηκε .git — παράλειψη hooks."
}

$venvPath = Join-Path $RepoRoot ".venv"

if (Test-Path $venvPath) {
    Write-Host "Το .venv υπάρχει ήδη."
} else {
    Write-Host "→ python -m venv .venv"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venvPath
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $venvPath
    } else {
        Write-Error "Δεν βρέθηκε Python. Εγκατάστησε Python 3 από python.org ή το Microsoft Store."
        exit 1
    }
    $py = Join-Path $venvPath "Scripts\python.exe"
    Write-Host "→ pip install --upgrade pip (μέσα στο venv)"
    & $py -m pip install --upgrade pip
    Write-Host "Έτοιμο: δημιουργήθηκε το .venv."
}

Write-Host ""
Write-Host "Ενεργοποίηση στο τρέχον τερματικό (PowerShell):"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "(cmd)  .venv\Scripts\activate.bat"
Write-Host "Αν μπλοκάρει το PowerShell: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
Write-Host "Απενεργοποίηση: deactivate"

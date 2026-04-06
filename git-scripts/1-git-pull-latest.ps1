# Ενημερώνει το τοπικό main από το origin (από ρίζα repo).
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

Write-Host "→ checkout main"
git checkout main

Write-Host "→ pull origin main"
git pull origin main

Write-Host "Έτοιμο: το main είναι ενημερωμένο."

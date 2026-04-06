# Συγχρονίζει το main και δημιουργεί νέο branch για task.
# Χρήση: .\scripts\git-start-task.ps1 feature\onoma-task
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Branch
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

Write-Host "→ checkout main"
git checkout main

Write-Host "→ pull origin main"
git pull origin main

Write-Host "→ νέο branch: $Branch"
git checkout -b $Branch

Write-Host "Έτοιμο: δουλεύεις στο branch «$Branch»."

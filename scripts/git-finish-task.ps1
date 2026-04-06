# Push του τρέχοντος branch (όχι main) στο origin με -u αν χρειάζεται.
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$current = (git rev-parse --abbrev-ref HEAD).Trim()
if ($current -eq "main" -or $current -eq "master") {
    Write-Host "Είσαι στο «$current». Πήγαινε πρώτα στο branch του task σου."
    exit 1
}

$dirty = git status --porcelain
if ($dirty) {
    Write-Host "Υπάρχουν μη-committed αλλαγές. Κάνε commit ή stash και ξανά."
    exit 1
}

Write-Host "→ push origin $current"
git push -u origin $current

Write-Host "Έτοιμο: στάλθηκε το «$current». Άνοιξε Pull request προς main στο GitHub."

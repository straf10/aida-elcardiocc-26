# Git — συνεργασία στο ELCardioCC

**Τι φτιάχνουμε (masterplan):** το **ELCardioCC 2026** είναι shared task (BioASQ/CLEF): σύστημα που παίρνει **ελληνικό κείμενο εξιτηρίου καρδιολογίας** και βγάζει **κωδικούς ICD-10** (πολυ-ετικέτα, 115 κωδικοί). Αξιολόγηση με **micro-F1** στο κρυφό test set· υποβολή JSONL/ZIP (έως 5 συστήματα). Εδώ περιγράφουμε το **Git workflow**: branch → **Pull request** στο GitHub → review → merge στο `main`.

---

## Πρώτη φορά

```bash
git clone <url-του-repo>
cd ELCardioCC
```

Προαιρετικά (ταυτότητα στα commits): `git config user.name "..."` και `git config user.email "..."`.

---

## Βοηθητικά scripts (εύκολη οδός)

Από τη **ρίζα του repo** (φάκελος `ELCardioCC`). Σε **Mac/Linux** πρώτη φορά: `chmod +x scripts/*.sh`.

| Τι | Mac / Linux | Windows (PowerShell στον φάκελο του repo) |
|----|-------------|-------------------------------------------|
| Τελευταίο `main` | `./scripts/git-pull-latest.sh` | `.\scripts\git-pull-latest.ps1` |
| Έναρξη task (νέο branch) | `./scripts/git-start-task.sh feature/onoma` | `.\scripts\git-start-task.ps1 feature/onoma` |
| Λήξη task (push branch) | `./scripts/git-finish-task.sh` | `.\scripts\git-finish-task.ps1` |

Το **finish** κάνει push μόνο αν **δεν** είσαι στο `main` και δεν υπάρχουν **αδιάθετες** αλλαγές (χωρίς commit). Μετά: άνοιξε **Pull request** προς `main` στο GitHub.

Αν το Windows μπλοκάρει scripts: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (μία φορά) ή `powershell -ExecutionPolicy Bypass -File .\scripts\git-pull-latest.ps1`.

---

## Βήματα με το χέρι (ίδια λογική με τα scripts)

1. `git checkout main` → `git pull origin main`  
2. `git checkout -b feature/paradeigma`  
3. δουλειά → `git add` → `git commit -m "..."`  
4. `git push -u origin feature/paradeigma` (μετά `git push`)  
5. PR στο GitHub· ο διαχειριστής κάνει merge.

Μετά το merge στο `main`: `git checkout main` → `git pull origin main` → `git branch -d feature/paradeigma`.

---

## Γρήγορη μνήμη

| Τι | Εντολή |
|----|--------|
| Νέο branch | Μετά `pull` στο `main`: `git checkout -b feature/...` |
| Κατάσταση | `git status` |
| Αποθήκευση | `git add` → `git commit -m "..."` |
| Πρώτο push branch | `git push -u origin feature/...` |

`git help <εντολή>` για λεπτομέρειες.

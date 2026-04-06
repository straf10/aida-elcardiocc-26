# Git — συνεργασία στο ELCardioCC

**Τι φτιάχνουμε (masterplan):** το **ELCardioCC 2026** είναι shared task (BioASQ/CLEF): σύστημα που παίρνει **ελληνικό κείμενο εξιτηρίου καρδιολογίας** και βγάζει **κωδικούς ICD-10** (πολυ-ετικέτα, 115 κωδικοί). Αξιολόγηση με **micro-F1** στο κρυφό test set· υποβολή JSONL/ZIP (έως 5 συστήματα). Εδώ περιγράφουμε το **Git workflow**: branch → **Pull request** στο GitHub → review → merge στο `main`.

---

## Πρώτη φορά

```bash
git clone <url-του-repo>
cd ELCardioCC
```

Προαιρετικά (ταυτότητα στα commits): `git config user.name "..."` και `git config user.email "..."`.

**Virtualenv (προτείνεται):** φτιάχνει `.venv` και στο τέλος εκτυπώνει πώς να το ενεργοποιήσεις. **macOS / Linux:** `chmod +x git-scripts/macos.sh` (μία φορά) και `./git-scripts/macos.sh`. **Windows (PowerShell):** `.\git-scripts\windows-setup.ps1`. Μετά τρέχεις τα παρακάτω με το `python` του venv.

---

## Βοηθητικά scripts (εύκολη οδός)

Από τη **ρίζα του repo**. Τα Git scripts είναι σε **Python 3** (ίδιο σε όλα τα OS)· χρειάζεσαι [Git](https://git-scm.com/) και Python.

| Τι | Εντολή |
|----|--------|
| Setup `.venv` | Mac/Linux: `./git-scripts/macos.sh` — Windows: `.\git-scripts\windows-setup.ps1` |
| Τελευταίο `main` | `python3 git-scripts/1-git-pull-latest.py` |
| Έναρξη task (νέο branch) | `python3 git-scripts/2-git-start-task.py feature/onoma` |
| Λήξη task (push branch) | `python3 git-scripts/3-git-finish-task.py` |

Το **finish** κάνει push μόνο αν **δεν** είσαι στο `main` και δεν υπάρχουν **αδιάθετες** αλλαγές (χωρίς commit). Μετά: **Pull request** προς `main` στο GitHub.

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

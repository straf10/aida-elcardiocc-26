# Git — συνεργασία στο ELCardioCC

**Τι φτιάχνουμε (masterplan):** το **ELCardioCC 2026** είναι shared task (BioASQ/CLEF): σύστημα που παίρνει **ελληνικό κείμενο εξιτηρίου καρδιολογίας** και βγάζει **κωδικούς ICD-10** (πολυ-ετικέτα, 115 κωδικοί). Αξιολόγηση με **micro-F1** στο κρυφό test set· υποβολή JSONL/ZIP (έως 5 συστήματα). **Git:** branch → **Pull request** στο GitHub → merge στο `main`.

```bash
git clone <url-του-repo>
cd ELCardioCC
```

Προαιρετικά: `git config user.name "..."` · `git config user.email "..."`

---

Όλες οι εντολές από τη **ρίζα** του repo. Φάκελος **`setup-scripts/`**:

**Βήμα 0** — Δημιουργεί `.venv`, ενημερώνει `pip`, εμφανίζει πώς να το ενεργοποιήσεις.

- macOS / Linux: `chmod +x setup-scripts/0-macos.sh` (μία φορά) · `./setup-scripts/0-macos.sh`
- Windows: `.\setup-scripts\0-windows.ps1`

**Βήμα 1** — Τραβά τις τελευταίες αλλαγές στο τοπικό `main` (`git checkout main` + `git pull`).

`python3 setup-scripts/1-git-pull-latest.py` (Windows: `python` ή `py -3` αντί για `python3`)

**Βήμα 2** — Συγχρονίζει το `main` και ανοίγει νέο branch για το task σου.

`python3 setup-scripts/2-git-start-task.py feature/onoma`

Μετά γράφεις κώδικα· για να «μπουν» στο branch χρειάζονται **`git add`** και **`git commit -m "..."`** (τα scripts δεν κάνουν commit μόνα τους).

**Βήμα 3** — Στέλνει στο GitHub το τρέχον branch (όχι αν είσαι στο `main` ή με αδιάθετες αλλαγές). Μετά: Pull request προς `main`.

`python3 setup-scripts/3-git-finish-task.py`

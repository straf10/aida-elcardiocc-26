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

## Φάκελος `setup-scripts/`

Όλα τρέχουν από τη **ρίζα του repo** (ο φάκελος `ELCardioCC` μετά το `cd`). Τα αρχεία είναι **αριθμημένα** κατά προτεινόμενη σειρά:

| # | Τι κάνει | macOS / Linux | Windows (PowerShell) |
|---|----------|---------------|----------------------|
| **0** | Φτιάχνει `.venv`, αναβαθμίζει `pip`, εκτυπώνει **activate** | `chmod +x setup-scripts/0-macos.sh` (μία φορά) · `./setup-scripts/0-macos.sh` | `.\setup-scripts\0-windows.ps1` |
| **1** | Ενημερώνει το τοπικό `main` από το GitHub | `python3 setup-scripts/1-git-pull-latest.py` | `python` ή `py -3` αντί για `python3` |
| **2** | `main` + νέο branch για task | `python3 setup-scripts/2-git-start-task.py feature/onoma` | ίδιο με κατάλληλο `python` |
| **3** | Push του τρέχοντος branch (όχι `main`) | `python3 setup-scripts/3-git-finish-task.py` | ίδιο |

**Ροή:** πρώτα το **0** (μία φορά ανά μηχάνημα). Μετά **ενεργοποίησε** το venv (`source .venv/bin/activate` ή `.\.venv\Scripts\Activate.ps1`) ώστε το `python` να δείχνει στο `.venv` — προαιρετικό αλλά προτείνεται. Στη συνέχεια χρησιμοποίησε **1 → 2** (ξεκίνα task), δούλεψε και κάνε `git add` / `git commit`, και στο τέλος **3** πριν ανοίξεις **Pull request** προς `main` στο GitHub.

Το **3** σταματά αν είσαι στο `main` ή αν υπάρχουν αδιάθετες αλλαγές χωρίς commit.

---

## Βήματα με το χέρι (ίδια λογική)

1. `git checkout main` → `git pull origin main`  
2. `git checkout -b feature/paradeigma`  
3. δουλειά → `git add` → `git commit -m "..."`  
4. `git push -u origin feature/paradeigma` (μετά `git push`)  
5. PR στο GitHub· ο διαχειριστής κάνει merge.

Μετά το merge: `git checkout main` → `git pull origin main` → `git branch -d feature/paradeigma`.

---

## Γρήγορη μνήμη

| Τι | Εντολή |
|----|--------|
| Νέο branch | Μετά `pull` στο `main`: `git checkout -b feature/...` |
| Κατάσταση | `git status` |
| Αποθήκευση | `git add` → `git commit -m "..."` |
| Πρώτο push branch | `git push -u origin feature/...` |

`git help <εντολή>` για λεπτομέρειες.

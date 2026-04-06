# Git — συνεργασία στο ELCardioCC

**Τι φτιάχνουμε (masterplan):** το **ELCardioCC 2026** είναι shared task (BioASQ/CLEF): σύστημα που παίρνει **ελληνικό κείμενο εξιτηρίου καρδιολογίας** και βγάζει **κωδικούς ICD-10** (πολυ-ετικέτα, 115 κωδικοί). Αξιολόγηση με **micro-F1** στο κρυφό test set· υποβολή JSONL/ZIP (έως 5 συστήματα).

**Ροή για τον χρήστη → admin**

| Φάση | Τι κάνεις εσύ | Τι γίνεται μετά |
|------|----------------|-------------------|
| **Νέο task** | Βήμα **2** — ανοίγεις branch για τη δουλειά σου | — |
| **Δουλειά** | Αλλαγές στο branch · **`git add`** · **`git commit`** (όσες φορές χρειάζεται) | — |
| **Finish** | Βήμα **3** — στέλνεις το branch στο GitHub | Άνοιξε **Pull request** προς `main` (σύνδεσμος από το GitHub) |
| **Review** | — | Ο **διαχειριστής** κάνει review και **merge** στο `main` (όχι εσύ, εκτός αν σου το επιτρέπουν) |

Πριν ξεκινήσεις **νέο** task μετά από merge: βήμα **1** ώστε το τοπικό σου `main` να είναι ενημερωμένο.

```bash
git clone <url-του-repo>
cd ELCardioCC
```

Προαιρετικά: `git config user.name "..."` · `git config user.email "..."`

---

Όλες οι εντολές από τη **ρίζα** του repo · φάκελος **`setup-scripts/`**:

**Βήμα 0** — Virtualenv (μία φορά ανά μηχάνημα).

- macOS / Linux: `chmod +x setup-scripts/0-macos.sh` · `./setup-scripts/0-macos.sh`
- Windows: `.\setup-scripts\0-windows.ps1`

**Βήμα 1** — Ενημέρωση τοπικού `main` από το GitHub.

`python3 setup-scripts/1-git-pull-latest.py` (Windows: `python` ή `py -3` αντί για `python3` — ίδιο και στα βήματα 2–3.)

**Βήμα 2** — **Νέο task:** `main` ενημερωμένο → νέο branch.

`python3 setup-scripts/2-git-start-task.py feature/onoma`

**Βήμα 3** — **Όταν είσαι έτοιμος:** push του branch (χωρίς αδιάθετες αλλαγές). Μετά άνοιξε PR — **review από admin**.

`python3 setup-scripts/3-git-finish-task.py`

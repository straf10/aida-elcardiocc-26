# Git — συνεργασία στο ELCardioCC

**Τι φτιάχνουμε (masterplan):** το **ELCardioCC 2026** είναι shared task (BioASQ/CLEF): σύστημα που παίρνει **ελληνικό κείμενο εξιτηρίου καρδιολογίας** και βγάζει **κωδικούς ICD-10** (πολυ-ετικέτα, 115 κωδικοί). Αξιολόγηση με **micro-F1** στο κρυφό test set· υποβολή JSONL/ZIP (έως 5 συστήματα). Εδώ περιγράφουμε μόνο το **Git workflow**: branch → δουλειά → **Pull/Merge request** → review από διαχειριστή → merge στο `main`.

---

## Πρώτη φορά

```bash
git clone <url-του-repo>
cd ELCardioCC
```

Προαιρετικά (ταυτότητα στα commits):

```bash
git config user.name "Το Όνομά σου"
git config user.email "email@example.com"
```

---

## Βήματα

**1. Ενημέρωσε το `main`**

```bash
git checkout main
git pull origin main
```

**2. Νέο branch** (π.χ. `feature/mlc-baseline`, `fix/preprocess`)

```bash
git checkout -b feature/paradeigma
```

**3. Αλλαγές και commit**

```bash
git status
git add path/αρχείου
git commit -m "Σύντομη περιγραφή"
```

**4. Push** (πρώτη φορά για το branch)

```bash
git push -u origin feature/paradeigma
```

Μετά: `git push`

**5. Αίτημα ελέγχου**

GitHub: **Pull requests** → νέο PR (`main` ← branch σου).  
GitLab: **Merge requests** → νέο MR.

Ο διαχειριστής κάνει review και merge στο `main` (εκτός αν έχεις δικαίωμα merge).

---

## Μετά το merge

```bash
git checkout main
git pull origin main
git branch -d feature/paradeigma
```

---

## Γρήγορη μνήμη

| Τι | Εντολή |
|----|--------|
| Νέο branch | Μετά `pull` στο `main`: `git checkout -b feature/...` |
| Κατάσταση | `git status` |
| Αποθήκευση | `git add` → `git commit -m "..."` |
| Πρώτο push branch | `git push -u origin feature/...` |

`git help <εντολή>` για λεπτομέρειες.

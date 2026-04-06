# Git — ELCardioCC

Δουλεύουμε πάνω σε **branch** (όχι απευθείας στο `main`). Ο καθένας φτιάχνει **δικό του branch** ανά task, π.χ. `feature/το-github-σου/σύντομο-όνομα`. Κάνεις αλλαγές, **commit**, στέλνεις στο GitHub, ανοίγεις **Pull request** — ο **διαχειριστής** κάνει merge.

Μετά το merge, πριν νέο task: τράβα το ενημερωμένο `main` (βήμα 1).

```bash
git clone <url-του-repo>
cd ELCardioCC
```

Προαιρετικά: `git config user.name "..."` · `git config user.email "..."`

---

Όλες οι εντολές από τη **ρίζα** του repo (`ELCardioCC`). Φάκελος **`setup-scripts/`**.

**0** — Φτιάχνει `.venv` (αν λείπει) και ενεργοποιεί τα **hooks** ώστε να μην κάνεις κατά λάθος commit στο `main`. Mac/Linux: `chmod +x setup-scripts/0-macos.sh` και `./setup-scripts/0-macos.sh`. Windows: `.\setup-scripts\0-windows.ps1`. Χωρίς βήμα 0: `git config core.hooksPath .githooks` μία φορά.

**1** — Φέρνει τοπικά τις τελευταίες αλλαγές στο `main` από το GitHub.  
`python3 setup-scripts/1-git-pull-latest.py` (στο Windows: `python` ή `py -3`)

**2** — Ξεκινάς **νέο task**: ενημερώνει το `main` και ανοίγει branch `feature/<username>/<όνομα-task>`. Τρέξε `python3 setup-scripts/2-git-start-task.py` και συμπλήρωσε username + όνομα όταν στα ζητήσει. (Προχωρημένα: ένα όρισμα, πλήρες όνομα branch.)

**3** — Στέλνει το branch στο GitHub (όχι αν έχεις αδιάθετες αλλαγές· πρώτα `git add` και `git commit`). Μετά άνοιξε **Pull request** — review από τον διαχειριστή.  
`python3 setup-scripts/3-git-finish-task.py`

Μεταξύ 2 και 3 δουλεύεις στο branch και κάνεις **όσα commits χρειάζεται** — τα scripts δεν κάνουν commit για σένα.

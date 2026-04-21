"""Blacklist, co-occurrence rules, and rule-based ICD boosts."""

from __future__ import annotations

from typing import Any

# Terms removed from the dictionary because they cause too many false positives.
BLACKLIST_TERMS = {
    "πε", "τr", "tr", "vt", "vf", "af", "as", "κμ", "πκμ", "αυ", "ay",
    "mr", "ar",
    "ονα", "απο", "οπο", "ολλ", "aor", "σελ",
    "ανιουσα αορτη",
    "μικρη mr",
    "ανεπαρκεια τριγλωχινας",
    "ανεπαρκεια τριγλωχινας βαλβιδας",
    "τριγλωχινα",
    "κα", "απ", "εμ", "πεμ", "α υ", "κ α", "ka", "acs",
}


def apply_cooccurrence_rules(
    predicted: set,
    rules: list[dict[str, Any]] | None = None,
) -> set:
    """
    Add highly co-occurring partner codes (training-derived).

    When ``rules`` is None, use the legacy hardcoded rules (I10/I11, I22→I21, I25→Z95).
    When ``rules`` is a non-empty list from YAML, each entry has ``if_any`` and ``then_add``.
    """
    if rules is None:
        if "I10" in predicted:
            predicted.add("I11")
        if "I11" in predicted:
            predicted.add("I10")
        if "I22" in predicted:
            predicted.add("I21")
        if "I25" in predicted:
            predicted.add("Z95")
        return predicted

    for r in rules:
        if_any = r.get("if_any") or []
        then_add = r.get("then_add") or []
        if any(c in predicted for c in if_any):
            predicted.update(then_add)
    return predicted


def apply_rule_based_boosts(
    text_norm: str,
    predicted: set,
    boosts: list[dict[str, Any]] | None = None,
    cooccurrence_rules: list[dict[str, Any]] | None = None,
) -> set:
    """
    Hard-coded clinical patterns that indicate specific ICD-10 codes.

    When ``boosts`` is None, use the legacy inline rules (same as pre-refactor).
    When ``boosts`` is provided (from YAML), match any pattern substring then add codes.
    Co-occurrence is applied at the end via ``apply_cooccurrence_rules`` with the same
    legacy or YAML list as ``cooccurrence_rules``.
    """
    if boosts is None:
        if any(x in text_norm for x in ["pci", "ppci", "ptca", "αγγειοπλαστικη"]):
            predicted.add("Y84")
        if any(x in text_norm for x in ["stent", "pacemaker", "ppm", "βηματοδοτη"]):
            predicted.add("Z95")
        if "tavi" in text_norm:
            predicted.update({"Y84", "Z95", "I35"})
        if "cabg" in text_norm:
            predicted.update({"Y84", "Z95", "I25"})

        if "μυοκαρδιτιδα" in text_norm:
            predicted.add("I41")
        if "περικαρδιτιδα" in text_norm:
            predicted.add("I30")
        if "πνευμονικη εμβολη" in text_norm:
            predicted.add("I26")
        if any(x in text_norm for x in ["κοιλιακη ταχυκαρδια", "κολπικη ταχυκαρδια"]):
            predicted.add("I47")
        if any(x in text_norm for x in ["κολποκοιλιακος αποκλεισμος", "mobitz ii"]):
            predicted.add("I44")
        if any(x in text_norm for x in ["οσς", "οξυ στεφανιαιο συνδρομο", "acs"]):
            predicted.update({"I20", "I21", "I22", "I24"})
        if any(x in text_norm for x in ["aov", "στενωση aov"]):
            predicted.add("I35")
        if "ανευρυσμα ανιουσας αορτης" in text_norm:
            predicted.add("I71")
        if "ισχαιμικο αεε" in text_norm:
            predicted.add("I64")
        if any(x in text_norm for x in ["καρδιακη ανακοπη", "αναταχθεισα ανακοπη"]):
            predicted.add("I46")

        if any(x in text_norm for x in ["οξεια νεφρικη βλαβη", "οξεια νεφρικη ανεπαρκεια"]):
            predicted.add("N17")
        if any(x in text_norm for x in ["αιμοκαθαρση", "τεχνητο νεφρο"]):
            predicted.add("Z99")
        if "λοιμωξη αναπνευστικου" in text_norm:
            predicted.add("J22")
        if "αναπνευστικη ανεπαρκεια" in text_norm:
            predicted.add("J96")

        if "ασκιτης" in text_norm:
            predicted.add("R18")
        if "αμυλοειδωσ" in text_norm:
            predicted.add("E85")
        if "takotsubo" in text_norm:
            predicted.add("I43")
    else:
        for boost in boosts:
            patterns = boost.get("patterns") or []
            add_codes = boost.get("add") or []
            if any(p in text_norm for p in patterns):
                predicted.update(add_codes)

    return apply_cooccurrence_rules(predicted, cooccurrence_rules)

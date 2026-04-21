"""Optional section and negation post-filters (P1; default off)."""

from __future__ import annotations

from typing import Any, Callable


def _normalize_header(h: str) -> str:
    from .normalize import normalize_text

    return normalize_text(h)


def _split_at_family_history(
    full_norm: str, family_headers_norm: list[str]
) -> tuple[str | None, str, str]:
    """Return (fh_segment_or_None, before, after) relative to first FH header match."""
    idx = -1
    found = None
    for h in family_headers_norm:
        if not h:
            continue
        pos = full_norm.find(h)
        if pos != -1 and (idx == -1 or pos < idx):
            idx = pos
            found = h
    if idx == -1:
        return None, full_norm, ""

    before = full_norm[:idx]
    after_header = full_norm[idx + len(found) :].strip()
    # Heuristic: FH block until next major section keyword (optional)
    stop_markers = [
        "διαγνωση",
        "φυσικη εξεταση",
        "εξετασεις",
        "θεραπεια",
        "εξιτήριο",
        "αιτιολογια εισαγωγης",
    ]
    end = len(after_header)
    for m in stop_markers:
        p = after_header.find(m)
        if p != -1:
            end = min(end, p)
    fh_segment = after_header[:end].strip()
    after = after_header[end:].strip()
    return fh_segment, before, after


def apply_section_filter(
    full_norm_text: str,
    predicted: set,
    cfg: dict[str, Any],
    match_dict_only: Callable[[str], set],
) -> set:
    """
    Optionally remove ICD codes that the dictionary matches only inside the family-history
    section (opt-in; default off).

    ``match_dict_only`` must return codes from substring dictionary matching only (no boosts).
    """
    if not cfg.get("enabled"):
        return predicted

    headers = cfg.get("family_history_headers") or []
    family_headers_norm = [_normalize_header(h) for h in headers if h]
    if not family_headers_norm:
        return predicted

    fh_seg, before, after = _split_at_family_history(full_norm_text, family_headers_norm)
    if fh_seg is None:
        return predicted

    rest_text = (before + " " + after).strip()
    pred_fh = match_dict_only(fh_seg)
    pred_rest = match_dict_only(rest_text)
    only_in_fh = pred_fh - pred_rest

    if cfg.get("drop_codes_in_family_history", True):
        predicted = set(predicted)
        predicted -= only_in_fh
        fh_code = cfg.get("family_history_code")
        if fh_code and fh_seg:
            predicted.add(fh_code)
    return predicted


def apply_negation_filter(
    full_norm_text: str,
    predicted: set,
    cfg: dict[str, Any],
) -> set:
    """
    Optionally remove codes when negation appears near a disease phrase (opt-in).

    YAML may include ``pairs: [{triggers: [negation, disease], remove: [...]}]``.
    Empty ``pairs`` → no change.
    """
    if not cfg.get("enabled"):
        return predicted

    pairs = cfg.get("pairs") or []
    if not pairs:
        return predicted

    predicted = set(predicted)
    tokens = full_norm_text.split()
    window = max(1, int(cfg.get("window_tokens", 4)))

    for pair in pairs:
        triggers = pair.get("triggers") or []
        remove_codes = set(pair.get("remove") or [])
        if len(triggers) < 2 or not remove_codes:
            continue
        neg, disease = triggers[0], triggers[1]
        if not neg or not disease:
            continue
        if neg not in full_norm_text or disease not in full_norm_text:
            continue
        try:
            i_neg = next(i for i, t in enumerate(tokens) if neg in t)
            i_dis = next(i for i, t in enumerate(tokens) if disease in t)
        except StopIteration:
            continue
        if abs(i_neg - i_dis) <= window + 1:
            predicted -= remove_codes
    return predicted

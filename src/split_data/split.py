import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import KFold


def labels_flat_from_record(rec: dict) -> set[str]:
    """Unique document-level ICD codes for one record (one count per code per document)."""
    doc_ann = rec.get("document_level_annotations") or []
    s: set[str] = set()
    for group in doc_ann:
        if isinstance(group, (list, tuple)):
            for code in group:
                if code is not None and str(code).strip():
                    s.add(str(code).strip())
    return s


def _pick_candidate(candidates: list[int], labels_flat_per_idx: list[set[str]]) -> int:
    """Break ties: prefer richer label sets for stratification stability."""
    return max(candidates, key=lambda i: (len(labels_flat_per_idx[i]), i))


def _apply_forced_rare_placements(
    label_to_docs: dict[str, list[int]],
    labels_flat_per_idx: list[set[str]],
) -> tuple[dict[int, str], list[dict]]:
    """
    Enforce per-label placement before iterative stratification:
    - count 1 -> train only
    - count 2-4 -> at least one train and one test
    - count >= 5 -> at least one train, val, and test
    """
    assignment: dict[int, str] = {}
    must_stay_train: set[int] = set()
    log: list[dict] = []

    labels_sorted = sorted(label_to_docs.keys(), key=lambda L: (len(label_to_docs[L]), L))

    for L in labels_sorted:
        docs = list(label_to_docs[L])
        c = len(docs)

        def cnt(split: str) -> int:
            return sum(1 for d in docs if assignment.get(d) == split)

        def unassigned() -> list[int]:
            return [d for d in docs if d not in assignment]

        if c == 1:
            d = docs[0]
            if d not in assignment:
                assignment[d] = "train"
                log.append({"label": L, "rule": "singleton", "idx": d, "split": "train"})
            elif assignment[d] != "train":
                raise ValueError(
                    f"Cannot place singleton label {L}: doc {d} already in {assignment[d]}"
                )
            must_stay_train.add(d)
            continue

        if 2 <= c <= 4:
            if cnt("train") == 0:
                cand = unassigned()
                if not cand:
                    raise ValueError(f"Label {L} (c={c}) needs train but no unassigned docs")
                d = _pick_candidate(cand, labels_flat_per_idx)
                assignment[d] = "train"
                log.append({"label": L, "rule": "2-4_train", "idx": d, "split": "train"})
            if cnt("test") == 0:
                cand = unassigned()
                if cand:
                    d = _pick_candidate(cand, labels_flat_per_idx)
                    assignment[d] = "test"
                    log.append({"label": L, "rule": "2-4_test", "idx": d, "split": "test"})
                else:
                    movable = [
                        d
                        for d in docs
                        if assignment.get(d) == "train" and d not in must_stay_train
                    ]
                    if not movable:
                        raise ValueError(
                            f"Label {L} (c={c}) needs test but cannot move any doc from train"
                        )
                    d = _pick_candidate(movable, labels_flat_per_idx)
                    assignment[d] = "test"
                    log.append(
                        {
                            "label": L,
                            "rule": "2-4_test_reassign",
                            "idx": d,
                            "to": "test",
                        }
                    )
            continue

        # c >= 5
        if cnt("train") == 0:
            cand = unassigned()
            if not cand:
                raise ValueError(f"Label {L} (c={c}) needs train but no unassigned docs")
            d = _pick_candidate(cand, labels_flat_per_idx)
            assignment[d] = "train"
            log.append({"label": L, "rule": "5+_train", "idx": d, "split": "train"})
        if cnt("val") == 0:
            cand = unassigned()
            if cand:
                d = _pick_candidate(cand, labels_flat_per_idx)
                assignment[d] = "val"
                log.append({"label": L, "rule": "5+_val", "idx": d, "split": "val"})
            else:
                movable = [
                    d
                    for d in docs
                    if assignment.get(d) == "test" and d not in must_stay_train
                ]
                if not movable:
                    movable = [
                        d
                        for d in docs
                        if assignment.get(d) == "train" and d not in must_stay_train
                    ]
                if not movable:
                    raise ValueError(f"Label {L} (c={c}) needs val but cannot place")
                d = _pick_candidate(movable, labels_flat_per_idx)
                assignment[d] = "val"
                log.append({"label": L, "rule": "5+_val_reassign", "idx": d, "to": "val"})
        if cnt("test") == 0:
            cand = unassigned()
            if cand:
                d = _pick_candidate(cand, labels_flat_per_idx)
                assignment[d] = "test"
                log.append({"label": L, "rule": "5+_test", "idx": d, "split": "test"})
            else:
                movable = [
                    d
                    for d in docs
                    if assignment.get(d) == "val" and d not in must_stay_train
                ]
                if not movable:
                    movable = [
                        d
                        for d in docs
                        if assignment.get(d) == "train" and d not in must_stay_train
                    ]
                if not movable:
                    raise ValueError(f"Label {L} (c={c}) needs test but cannot place")
                d = _pick_candidate(movable, labels_flat_per_idx)
                assignment[d] = "test"
                log.append({"label": L, "rule": "5+_test_reassign", "idx": d, "to": "test"})

    return assignment, log


def _reconcile_test_size(
    pool: np.ndarray,
    test_idx: np.ndarray,
    train_val_idx: np.ndarray,
    need_test: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure exact test count after MSSS (off-by-one from rounding)."""
    test_idx = np.asarray(test_idx)
    train_val_idx = np.asarray(train_val_idx)
    got = len(test_idx)
    if got == need_test:
        return test_idx, train_val_idx
    pool_set = set(pool.tolist())
    if got > need_test:
        extra = test_idx[need_test:]
        test_idx = test_idx[:need_test]
        train_val_idx = np.unique(np.concatenate([train_val_idx, extra]))
    else:
        need_more = need_test - got
        take = train_val_idx[:need_more]
        train_val_idx = train_val_idx[need_more:]
        test_idx = np.unique(np.concatenate([test_idx, take]))
    assert set(test_idx.tolist()) | set(train_val_idx.tolist()) == pool_set
    assert len(test_idx) == need_test
    assert len(train_val_idx) == len(pool) - need_test
    return test_idx, train_val_idx


def _reconcile_val_size(
    train_val_pool: np.ndarray,
    val_idx: np.ndarray,
    train_idx: np.ndarray,
    need_val: int,
) -> tuple[np.ndarray, np.ndarray]:
    val_idx = np.asarray(val_idx)
    train_idx = np.asarray(train_idx)
    got = len(val_idx)
    if got == need_val:
        return val_idx, train_idx
    full = set(train_val_pool.tolist())
    if got > need_val:
        extra = val_idx[need_val:]
        val_idx = val_idx[:need_val]
        train_idx = np.unique(np.concatenate([train_idx, extra]))
    else:
        need_more = need_val - got
        take = train_idx[:need_more]
        train_idx = train_idx[need_more:]
        val_idx = np.unique(np.concatenate([val_idx, take]))
    assert set(val_idx.tolist()) | set(train_idx.tolist()) == full
    assert len(val_idx) == need_val
    assert len(train_idx) == len(train_val_pool) - need_val
    return val_idx, train_idx


def _patch_missing_train_labels(
    train_set: set[int],
    val_set: set[int],
    test_set: set[int],
    label_to_docs: dict[str, list[int]],
    labels_flat_per_idx: list[set[str]],
) -> list[dict]:
    """After MSSS, ensure every label appears at least once in train."""
    patches: list[dict] = []
    train_labels: set[str] = set()
    for i in train_set:
        train_labels |= labels_flat_per_idx[i]

    for L in sorted(label_to_docs.keys()):
        if L in train_labels:
            continue
        cand = [i for i in sorted(val_set) if L in labels_flat_per_idx[i]]
        src = "val"
        if not cand:
            cand = [i for i in sorted(test_set) if L in labels_flat_per_idx[i]]
            src = "test"
        if not cand:
            raise ValueError(f"Label {L} missing in train and no donor in val/test")
        i = cand[0]
        val_set.discard(i)
        test_set.discard(i)
        train_set.add(i)
        patches.append({"label": L, "idx": i, "from": src})
        train_labels.add(L)
    return patches


def _verify_rare_label_rules(
    label_to_docs: dict[str, list[int]],
    train_set: set[int],
    val_set: set[int],
    test_set: set[int],
) -> None:
    for L, docs in label_to_docs.items():
        c = len(docs)
        in_t = sum(1 for d in docs if d in train_set)
        in_v = sum(1 for d in docs if d in val_set)
        in_s = sum(1 for d in docs if d in test_set)
        if c == 1:
            if in_t != 1:
                raise AssertionError(f"Singleton {L}: expected 1 in train, got {in_t}")
        elif 2 <= c <= 4:
            if in_t < 1 or in_s < 1:
                raise AssertionError(
                    f"Label {L} (c={c}): need train and test, got train={in_t} test={in_s}"
                )
        else:
            if in_t < 1 or in_v < 1 or in_s < 1:
                raise AssertionError(
                    f"Label {L} (c={c}): need train/val/test, got {in_t}/{in_v}/{in_s}"
                )


def _build_label_report(
    label_to_docs: dict[str, list[int]],
    train_set: set[int],
    val_set: set[int],
    test_set: set[int],
) -> dict[str, dict]:
    per_label: dict[str, dict] = {}
    for L in sorted(label_to_docs.keys()):
        docs = label_to_docs[L]
        per_label[L] = {
            "count_total": len(docs),
            "count_train": sum(1 for d in docs if d in train_set),
            "count_val": sum(1 for d in docs if d in val_set),
            "count_test": sum(1 for d in docs if d in test_set),
        }
    return per_label


def stratified_train_val_test_split(
    records: list[dict],
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """
    80/10/10-style multilabel stratified split (adjust via val_size, test_size).
    Enforces rare-label placement rules, then two-stage MultilabelStratifiedShuffleSplit
    on the remaining pool. Guarantees every label with count>=1 appears in train.

    Returns (train_records, val_records, test_records, report_dict).
    Records are shallow copies without ``labels_flat`` / ``_row_id`` (committee schema only).
    """
    n = len(records)
    if n == 0:
        return [], [], [], {"error": "empty records"}

    train_target = int(round((1.0 - val_size - test_size) * n))
    val_target = int(round(val_size * n))
    test_target = n - train_target - val_target

    labels_flat_per_idx = [labels_flat_from_record(records[i]) for i in range(n)]

    label_to_docs: dict[str, list[int]] = defaultdict(list)
    for i, lf in enumerate(labels_flat_per_idx):
        for L in lf:
            label_to_docs[L].append(i)
    label_to_docs = {k: sorted(set(v)) for k, v in label_to_docs.items()}

    forced_assignment, forced_log = _apply_forced_rare_placements(
        label_to_docs, labels_flat_per_idx
    )

    forced_train = {i for i, s in forced_assignment.items() if s == "train"}
    forced_val = {i for i, s in forced_assignment.items() if s == "val"}
    forced_test = {i for i, s in forced_assignment.items() if s == "test"}

    need_train = train_target - len(forced_train)
    need_val = val_target - len(forced_val)
    need_test = test_target - len(forced_test)

    if need_train < 0 or need_val < 0 or need_test < 0:
        raise ValueError(
            f"Forced placements exceed targets: need_train={need_train}, "
            f"need_val={need_val}, need_test={need_test}"
        )

    pool = [i for i in range(n) if i not in forced_assignment]
    if need_train + need_val + need_test != len(pool):
        raise ValueError(
            f"Pool size mismatch: pool={len(pool)}, "
            f"need_train+val+test={need_train + need_val + need_test}"
        )

    train_set: set[int] = set(forced_train)
    val_set: set[int] = set(forced_val)
    test_set: set[int] = set(forced_test)

    if pool:
        pool_arr = np.array(pool, dtype=np.int64)
        pool_labels = [sorted(labels_flat_per_idx[i]) for i in pool]

        mlb_a = MultiLabelBinarizer()
        Y_a = mlb_a.fit_transform(pool_labels)
        X_a = np.arange(len(pool)).reshape(-1, 1)

        t_frac = need_test / len(pool) if len(pool) else 0.0
        msss_a = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=t_frac, random_state=seed
        )
        test_local: np.ndarray | None = None
        tv_local: np.ndarray | None = None
        for tr_loc, te_loc in msss_a.split(X_a, Y_a):
            test_local = te_loc
            tv_local = tr_loc
        assert test_local is not None and tv_local is not None

        test_from_pool = pool_arr[test_local]
        train_val_from_pool = pool_arr[tv_local]

        test_from_pool, train_val_from_pool = _reconcile_test_size(
            pool_arr, test_from_pool, train_val_from_pool, need_test
        )

        tv_labels = [sorted(labels_flat_per_idx[i]) for i in train_val_from_pool.tolist()]
        mlb_b = MultiLabelBinarizer()
        Y_b = mlb_b.fit_transform(tv_labels)
        X_b = np.arange(len(train_val_from_pool)).reshape(-1, 1)

        v_frac = need_val / len(train_val_from_pool) if len(train_val_from_pool) else 0.0
        msss_b = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=v_frac, random_state=seed + 1
        )
        val_local: np.ndarray | None = None
        tr_local: np.ndarray | None = None
        for tr_loc, va_loc in msss_b.split(X_b, Y_b):
            val_local = va_loc
            tr_local = tr_loc
        assert val_local is not None and tr_local is not None

        val_from_pool = train_val_from_pool[val_local]
        train_from_pool = train_val_from_pool[tr_local]

        val_from_pool, train_from_pool = _reconcile_val_size(
            train_val_from_pool, val_from_pool, train_from_pool, need_val
        )

        train_set.update(train_from_pool.tolist())
        val_set.update(val_from_pool.tolist())
        test_set.update(test_from_pool.tolist())

    patch_log = _patch_missing_train_labels(
        train_set, val_set, test_set, label_to_docs, labels_flat_per_idx
    )

    _verify_rare_label_rules(label_to_docs, train_set, val_set, test_set)

    if train_set | val_set | test_set != set(range(n)):
        raise ValueError("Split does not partition indices")
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Overlapping splits")

    def export_record(rec: dict) -> dict:
        out: dict = {
            "patient_id": rec["patient_id"],
            "text": rec["text"],
            "document_level_annotations": rec.get("document_level_annotations", []),
        }
        m = rec.get("mention_level_annotations")
        if m:
            out["mention_level_annotations"] = m
        return out

    train_list = [export_record(records[i]) for i in sorted(train_set)]
    val_list = [export_record(records[i]) for i in sorted(val_set)]
    test_list = [export_record(records[i]) for i in sorted(test_set)]

    np.random.RandomState(seed).shuffle(train_list)
    np.random.RandomState(seed + 2).shuffle(val_list)
    np.random.RandomState(seed + 3).shuffle(test_list)

    train_labels_final = set()
    for i in train_set:
        train_labels_final |= labels_flat_per_idx[i]
    labels_zero_in_val = [L for L in label_to_docs if sum(1 for d in label_to_docs[L] if d in val_set) == 0]
    labels_zero_in_test = [L for L in label_to_docs if sum(1 for d in label_to_docs[L] if d in test_set) == 0]

    report = {
        "n_total": n,
        "train_target": train_target,
        "val_target": val_target,
        "test_target": test_target,
        "actual_train": len(train_list),
        "actual_val": len(val_list),
        "actual_test": len(test_list),
        "n_labels_present": len(label_to_docs),
        "labels_with_zero_in_val": labels_zero_in_val,
        "labels_with_zero_in_test": labels_zero_in_test,
        "forced_placements": forced_log,
        "train_coverage_patches": patch_log,
        "per_label": _build_label_report(label_to_docs, train_set, val_set, test_set),
    }

    if labels_zero_in_val:
        print(f"WARNING: {len(labels_zero_in_val)} labels have zero samples in val (ok by spec).")
    if labels_zero_in_test:
        print(f"WARNING: {len(labels_zero_in_test)} labels have zero samples in test (ok by spec).")

    missing_in_train = set(label_to_docs.keys()) - train_labels_final
    if missing_in_train:
        raise ValueError(f"Labels still missing from train after patch: {missing_in_train}")

    print(
        f"3-way split OK: train={len(train_list)} val={len(val_list)} test={len(test_list)} "
        f"(targets {train_target}/{val_target}/{test_target})"
    )
    print(f"Forced placements: {len(forced_log)}, train patches: {len(patch_log)}")

    return train_list, val_list, test_list, report


def ensure_train_coverage(data: list[dict]) -> tuple[set[int], set[str]]:
    """
    Identify samples that must go to train to ensure val labels are covered.
    Returns: (forced_train_indices, singleton_labels)
    """
    label_to_samples = defaultdict(set)
    for idx, sample in enumerate(data):
        for label in sample["labels_flat"]:
            label_to_samples[label].add(idx)

    # Singleton labels (count=1) must go to train only
    singleton_labels = {lbl for lbl, idxs in label_to_samples.items() if len(idxs) == 1}

    forced_train = set()
    for label in singleton_labels:
        forced_train.update(label_to_samples[label])

    # Rare labels (count 2-5): ensure at least 1 in train
    rare_labels = {lbl for lbl, idxs in label_to_samples.items() if 2 <= len(idxs) <= 5}
    for label in rare_labels:
        sample_idxs = list(label_to_samples[label])
        if not any(i in forced_train for i in sample_idxs):
            forced_train.add(sample_idxs[0])

    return forced_train, singleton_labels


def validate_split(train_data_list: list[dict], val_data_list: list[dict]) -> None:
    """Ensure every validation label exists in training."""
    train_labels = set()
    for sample in train_data_list:
        train_labels.update(sample["labels_flat"])

    val_labels = set()
    for sample in val_data_list:
        val_labels.update(sample["labels_flat"])

    val_only = val_labels - train_labels
    if val_only:
        raise ValueError(f"Validation contains labels not in train: {sorted(val_only)}")

    print(f"Split validation passed. Train-only labels: {len(train_labels - val_labels)}")


def stratified_train_val_split(records: list[dict], test_size: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    MultilabelStratifiedShuffleSplit + forced-train coverage for singleton/rare labels.
    Splits on `labels_flat` (computed from `document_level_annotations`).
    """
    forced_train_idx, singleton_labels = ensure_train_coverage(records)

    if singleton_labels:
        print(f"\nWARNING: Found {len(singleton_labels)} singleton labels (count=1).")
        print(f"These will be forced to the training set to prevent validation-only labels.")

    forced_train_data = [records[i] for i in forced_train_idx]
    pool_data = [records[i] for i in range(len(records)) if i not in forced_train_idx]

    # Perform stratified split on the remaining pool
    mlb = MultiLabelBinarizer()
    labels_list = [x["labels_flat"] for x in pool_data]

    if labels_list:
        Y = mlb.fit_transform(labels_list)
        X = np.arange(len(pool_data)).reshape(-1, 1)

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

        train_data = []
        val_data = []
        for train_idx, val_idx in msss.split(X, Y):
            train_data = [pool_data[i] for i in train_idx]
            val_data = [pool_data[i] for i in val_idx]
    else:
        train_data = []
        val_data = []

    # Merge forced train samples back into train_data
    train_data.extend(forced_train_data)

    # Shuffle the final train_data to mix in the forced samples
    np.random.seed(seed)
    np.random.shuffle(train_data)

    # Validate the split
    validate_split(train_data, val_data)

    return train_data, val_data


def make_kfold_splits(records: list[dict], n_splits: int = 5, seed: int = 42) -> list[tuple[list[dict], list[dict]]]:
    """
    For xlm_r_base. Same records list, standard k-fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(records):
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]
        splits.append((train_recs, val_recs))
    return splits

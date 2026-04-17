import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import KFold


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

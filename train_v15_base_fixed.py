import argparse, csv, json, os, random, re, time
import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging

transformers_logging.set_verbosity_error()
random.seed(42)
np.random.seed(42)

def load_data(path):
    with open(path, encoding='utf-8') as f: return [json.loads(l) for l in f]

def load_labelset(path):
    with open(path, encoding='utf-8') as f: return [l.strip() for l in f if l.strip()]

def load_label_descriptions(csv_path, labelset):
    desc = {}
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            desc[row['code']] = row['greek_description']
    return [desc.get(l, l) for l in labelset]

def load_synonym_dict(csv_path):
    code_to_terms = defaultdict(list)
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            term = row['term'].strip()
            for code in row['codes_pipe_sep'].split('|'):
                code = code.strip()
                if term: code_to_terms[code].append(term)
    return code_to_terms

def flatten_labels(r):
    return list({c for g in r['document_level_annotations'] for c in g})

def synonym_augment(text: str, rec_codes: list, code_to_terms: dict, p: float = 0.35) -> str:
    result = text
    for code in rec_codes:
        terms = code_to_terms.get(code, [])
        if len(terms) < 2: continue
        for term in terms:
            if random.random() > p: continue
            if len(term) < 4: continue
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(result):
                replacement = random.choice([t for t in terms if t != term and len(t) >= 4])
                if replacement: result = pattern.sub(replacement, result, count=1)
                break
    return result

def build_augmented_dataset(records: list, labelset: list, code_to_terms: dict) -> list:
    label_freq = Counter(c for r in records for g in r['document_level_annotations'] for c in g)
    aug_mult = {}
    for code in labelset:
        freq = label_freq.get(code, 0)
        if   freq < 30:   aug_mult[code] = 0  
        elif freq < 100:  aug_mult[code] = 4
        elif freq < 200:  aug_mult[code] = 3
        elif freq < 300:  aug_mult[code] = 2
        else:             aug_mult[code] = 0

    augmented = list(records)
    for record in records:
        rec_codes  = flatten_labels(record)
        multiplier = max((aug_mult.get(c, 0) for c in rec_codes), default=0)
        for _ in range(multiplier):
            new_text = synonym_augment(record['text'], rec_codes, code_to_terms)
            augmented.append({
                'patient_id':                 f"{record['patient_id']}_aug",
                'document_level_annotations': record['document_level_annotations'],
                'mention_level_annotations':  record.get('mention_level_annotations', []),
                'text':                       new_text,
            })
    random.shuffle(augmented)
    return augmented

def build_cooccurrence_rules(records, min_cooccur=40, min_prob=0.45):
    freq      = Counter(c for r in records for g in r['document_level_annotations'] for c in g)
    pair_freq = Counter()
    for r in records:
        codes = sorted(set(c for g in r['document_level_annotations'] for c in g))
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                pair_freq[(codes[i], codes[j])] += 1
    rules = {}
    for (a, b), co in pair_freq.items():
        if co >= min_cooccur:
            if co / max(freq[a], 1) >= min_prob: rules.setdefault(a, []).append((b, round(co / freq[a], 3)))
            if co / max(freq[b], 1) >= min_prob: rules.setdefault(b, []).append((a, round(co / freq[b], 3)))
    return rules

class ClinicalDataset(Dataset):
    def __init__(self, records, tokenizer, mlb, max_len=512):
        self.records   = records
        self.tokenizer = tokenizer
        self.labels    = mlb.transform([flatten_labels(r) for r in records]).astype(np.float32)
        self.max_len   = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.records[idx]['text'], max_length=self.max_len, truncation=True,
            truncation_side='left', padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.float),
        }

class MedicalModelWithDescriptions(nn.Module):
    def __init__(self, model_name: str, num_labels: int, label_descriptions: list, tokenizer, device):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        H               = self.encoder.config.hidden_size
        self.dropouts   = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])
        self.classifier = nn.Linear(H, num_labels)
        self.alpha      = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('desc_emb', self._encode_descriptions(label_descriptions, tokenizer, device, model_name))

    @torch.no_grad()
    def _encode_descriptions(self, descriptions, tokenizer, device, model_name):
        temp_encoder = AutoModel.from_pretrained(model_name).to(device)
        temp_encoder.eval()
        all_cls = []
        for desc in descriptions:
            enc = tokenizer(desc, max_length=64, truncation=True, padding='max_length', return_tensors='pt').to(device)
            out = temp_encoder(**enc)
            all_cls.append(out.last_hidden_state[:, 0, :].cpu())
        del temp_encoder
        torch.cuda.empty_cache()
        emb = torch.cat(all_cls, dim=0)
        return nn.functional.normalize(emb, dim=-1)

    def forward(self, input_ids, attention_mask):
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        doc_cls = out.last_hidden_state[:, 0, :]
        base_logits = sum(self.classifier(dp(doc_cls)) for dp in self.dropouts) / len(self.dropouts)
        doc_cls_norm = nn.functional.normalize(doc_cls, dim=-1)
        desc_sim     = doc_cls_norm @ self.desc_emb.T
        return base_logits + self.alpha * desc_sim

def compute_pos_weights(train_records, labelset, cap=20.0):
    n      = len(train_records)
    l2i    = {l: i for i, l in enumerate(labelset)}
    counts = np.zeros(len(labelset), dtype=np.float32)
    for r in train_records:
        for code in flatten_labels(r):
            if code in l2i: counts[l2i[code]] += 1
    pw = (n - counts) / np.maximum(counts, 1.0)
    return torch.tensor(np.clip(pw, 1.0, cap), dtype=torch.float)

@torch.no_grad()
def evaluate(model, loader, device, thresholds=None, n_labels=115):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        lg = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        all_logits.append(lg.cpu())
        all_labels.append(batch['labels'])
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)
    proba  = torch.sigmoid(torch.tensor(logits)).numpy()
    if thresholds is None: thresholds = np.full(n_labels, 0.5)
    y_pred = (proba >= thresholds).astype(int)
    f1     = f1_score(y_true, y_pred, average='micro', zero_division=0)
    avg_p  = y_pred.sum(axis=1).mean()
    return f1, proba, y_true, avg_p

def tune_thresholds(proba, y_true, labelset):
    thresholds = np.full(len(labelset), 0.5)
    for j in range(len(labelset)):
        col = proba[:, j]
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 50):
            f1 = f1_score(y_true[:, j], (col >= t).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t
        thresholds[j] = best_t
    return thresholds

ICD_HIERARCHY = {'I11': 'I10', 'I22': 'I21'}

def apply_icd_hierarchy(y_pred: np.ndarray, labelset: list) -> np.ndarray:
    l2i    = {l: i for i, l in enumerate(labelset)}
    result = y_pred.copy()
    for child, parent in ICD_HIERARCHY.items():
        if child not in l2i or parent not in l2i: continue
        ci, pi = l2i[child], l2i[parent]
        mask = (result[:, ci] == 1) & (result[:, pi] == 0)
        result[mask, pi] = 1
    return result

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    records      = load_data(args.data)
    labelset     = load_labelset(args.labels)
    mlb          = MultiLabelBinarizer(classes=labelset).fit([labelset])
    tokenizer    = AutoTokenizer.from_pretrained('xlm-roberta-base')
    label_descs  = load_label_descriptions(args.desc_csv, labelset)
    code_to_terms = load_synonym_dict(args.syn_csv)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(records), 1):
        print(f'\n=== FOLD {fold}/{args.folds} ===')
        train_recs = [records[i] for i in train_idx]
        val_recs   = [records[i] for i in val_idx]

        aug_train = build_augmented_dataset(train_recs, labelset, code_to_terms)

        train_ds = ClinicalDataset(aug_train, tokenizer, mlb, args.max_len)
        val_ds   = ClinicalDataset(val_recs,  tokenizer, mlb, args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2, shuffle=False, num_workers=2, pin_memory=True)

        model = MedicalModelWithDescriptions('xlm-roberta-base', len(labelset), label_descs, tokenizer, device).to(device)

        pw      = compute_pos_weights(aug_train, labelset, cap=20.0).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        best_f1, best_proba, best_ytrue, patience_ctr = 0.0, None, None, 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            tot_loss = 0.0
            t0       = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                loss   = loss_fn(logits, batch['labels'].to(device))
                (loss / args.grad_acc).backward()
                tot_loss += loss.item()

                if (step + 1) % args.grad_acc == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            f1, proba, ytrue, avg_p = evaluate(model, val_loader, device, n_labels=len(labelset))
            print(f'  Fold {fold} Ep {epoch:2d} | loss={tot_loss/len(train_loader):.4f} | F1={f1:.4f} | {time.time()-t0:.0f}s')

            if f1 > best_f1:
                best_f1, best_proba, best_ytrue, patience_ctr = f1, proba.copy(), ytrue.copy(), 0
                torch.save(model.state_dict(), os.path.join(args.out, f'model_fold_{fold-1}.pt'))
                print(f'    ★ Saved (F1={best_f1:.4f}) | alpha={model.alpha.item():.4f}')
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience: break

        thresholds = tune_thresholds(best_proba, best_ytrue, labelset)
        y_pred_raw  = (best_proba >= thresholds).astype(int)
        f1_hier     = f1_score(best_ytrue, apply_icd_hierarchy(y_pred_raw, labelset), average='micro', zero_division=0)
        results.append({'fold': fold, 'thresholds': thresholds, 'best_f1': best_f1, 'f1_tuned': f1_hier})

    np.save(os.path.join(args.out, 'avg_thresholds.npy'), np.mean([r['thresholds'] for r in results], axis=0))
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, 'icd_hierarchy.json'), 'w') as f: json.dump(ICD_HIERARCHY, f)
    
    # -------------------------------------------------------------------------
    # Μετά τον fold loop — rules για inference από ΟΛΟΚΛΗΡΟ το dataset
    # -------------------------------------------------------------------------
    print('\nBuilding final co-occurrence rules from the training dataset for inference...', flush=True)
    co_rules_full = build_cooccurrence_rules(records)
    with open(os.path.join(args.out, 'cooccurrence_rules.json'), 'w', encoding='utf-8') as f:
        json.dump(co_rules_full, f, ensure_ascii=False, indent=2)

    print(f'\nMean F1 (tuned + hierarchy): {np.mean([r["f1_tuned"] for r in results]):.4f}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # Αλλαγή στο training_set.jsonl για πλήρη διαχωρισμό!
    p.add_argument('--data',      default='training_set.jsonl')
    p.add_argument('--labels',    default='labelset.txt')
    p.add_argument('--desc_csv',  default='icd10_greek_lookup.csv')
    p.add_argument('--syn_csv',   default='full_dictionary.csv')
    p.add_argument('--out',       default='model_v15_base_fixed')
    p.add_argument('--folds',     type=int,   default=5)
    p.add_argument('--epochs',    type=int,   default=20)
    p.add_argument('--batch',     type=int,   default=12)
    p.add_argument('--grad_acc',  type=int,   default=1)
    p.add_argument('--lr',        type=float, default=2e-5)
    p.add_argument('--max_len',   type=int,   default=512)
    p.add_argument('--patience',  type=int,   default=4)
    main(p.parse_args())

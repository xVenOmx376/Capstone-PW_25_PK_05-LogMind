"""STEP 4: statistical engine = Isolation Forest fitted on NORMAL training sessions only.
score = fraction of normal training sessions that look LESS anomalous (0-1, higher = rarer);
threshold = 95th percentile of training scores (unsupervised - not tuned on test labels).
Expected: AUROC clearly above 0.5 (healthy pipeline: ~0.8+). Modest recall is fine for a baseline."""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from config import *
from common import *

if not SESSIONS_PKL.exists():
    sys.exit("Run step3_build_sessions.py first.")
S, vocab, tm = load_sessions(), load_vocab(), load_templates()
train, test = S[S.split == "train"].reset_index(drop=True), S[S.split == "test"].reset_index(drop=True)
Xtr, Ctr = feature_matrix(train, vocab)
Xte, Cte = feature_matrix(test, vocab)

model = IsolationForest(n_estimators=200, random_state=SEED).fit(Xtr)
raw_tr = np.sort(-model.score_samples(Xtr))                      # higher = more anomalous
raw_te = -model.score_samples(Xte)
score = np.searchsorted(raw_tr, raw_te) / len(raw_tr)            # percentile vs normal training sessions
pred = (score >= STAT_THRESHOLD).astype(int)

mean, frac_present = Ctr.mean(0), (Ctr > 0).mean(0)
z = (Cte - mean) / (Ctr.std(0) + 0.1)


def evidence(i):
    parts = []
    for j in np.argsort(-np.abs(z[i]))[:2]:
        t = tm[vocab[j]]
        if abs(z[i, j]) < 1: continue
        if Cte[i, j] == 0: parts.append(f'missing "{t[:70]}" (present in {frac_present[j]:.0%} of normal sessions)')
        else: parts.append(f'{int(Cte[i, j])}x "{t[:70]}" (normal avg {mean[j]:.2f})')
    return "; ".join(parts) or "no single template stands out"


out = pd.DataFrame({"block_id": test.block_id, "label": test.label, "stat_score": score.round(3),
                    "stat_decision": np.where(pred == 1, "Anomalous", "Normal"),
                    "evidence": [evidence(i) for i in range(len(test))]})
out.to_csv(STAT_CSV, index=False)
p, r, f, _ = precision_recall_fscore_support(test.label, pred, average="binary", zero_division=0)
auc = roc_auc_score(test.label, score)
m = {"precision": p, "recall": r, "f1": f, "auroc": auc, "threshold": STAT_THRESHOLD, "n_test": len(test)}
json.dump(m, open(STAT_METRICS, "w"), indent=1)
print(f"Statistical engine on {len(test)} test sessions (threshold {STAT_THRESHOLD}):")
print(f"  precision {p:.3f} | recall {r:.3f} | F1 {f:.3f} | AUROC {auc:.3f}")
if auc < 0.6: print("  WARNING: AUROC near 0.5 - usually a bug (train/test column mismatch).")

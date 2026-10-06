"""STEP 1: sample sessions (blocks) from HDFS and keep only their log lines.
Expected: 4000 train-normal / 100 test-normal / 100 test-anomaly blocks, no overlap, ~80k-120k lines kept."""
import random, sys
import pandas as pd
from config import *

if not RAW_LOG.exists() or not RAW_LABELS.exists():
    sys.exit(f"Missing {RAW_LOG.name} / {RAW_LABELS.name} in data/. Download HDFS_v1 from LogHub "
             "(or run make_synthetic_data.py for a dry run).")

lab = pd.read_csv(RAW_LABELS)
lab.columns = [c.strip() for c in lab.columns]
normal = sorted(lab.loc[lab.Label == "Normal", "BlockId"])
anom = sorted(lab.loc[lab.Label == "Anomaly", "BlockId"])
print(f"Full label file: {len(lab)} blocks, {len(anom)} anomalous ({len(anom)/len(lab):.1%})")

rng = random.Random(SEED)
rng.shuffle(normal); rng.shuffle(anom)
need = N_TRAIN + N_TEST_NORMAL
if len(normal) < need or len(anom) < N_TEST_ANOMALY:
    sys.exit(f"Not enough blocks: have {len(normal)} normal / {len(anom)} anomalous, need {need} / {N_TEST_ANOMALY}")

split = {}
for b in normal[:N_TRAIN]: split[b] = ("train", 0)
for b in normal[N_TRAIN:need]: split[b] = ("test", 0)
for b in anom[:N_TEST_ANOMALY]: split[b] = ("test", 1)

pd.DataFrame([(b, s, l) for b, (s, l) in split.items()], columns=["block_id", "split", "label"]).to_csv(SPLIT_CSV, index=False)

kept = 0
with open(RAW_LOG, errors="replace") as fin, open(SAMPLE_LOG, "w") as fout:
    for line in fin:
        if any(b in split for b in BLOCK_RE.findall(line)):
            fout.write(line); kept += 1

df = pd.read_csv(SPLIT_CSV)
print(df.groupby(["split", "label"]).size().rename("blocks").to_string())
assert df.block_id.is_unique, "a block appears in two splits!"
print(f"Kept {kept} log lines -> {SAMPLE_LOG.name}; split table -> {SPLIT_CSV.name}")

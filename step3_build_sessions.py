"""STEP 3: group parsed lines by block ID into sessions.
Expected: ~4,200 sessions (4000 train / 200 test), avg ~20 lines; anomalous sessions contain rarer templates or odd lengths."""
import json, sys
from collections import defaultdict
import pandas as pd
from config import *

if not PARSED_CSV.exists():
    sys.exit("Run step2_parse_drain.py first.")
parsed = pd.read_csv(PARSED_CSV, dtype={"blocks": str}).fillna({"blocks": ""})
split = pd.read_csv(SPLIT_CSV).set_index("block_id")

acc = defaultdict(lambda: {"seq": [], "line_idx": [], "ts": []})
for r in parsed.itertuples(index=False):
    for b in r.blocks.split("|"):
        if b in split.index:
            acc[b]["seq"].append(int(r.template_id)); acc[b]["line_idx"].append(int(r.line_idx)); acc[b]["ts"].append(r.timestamp)

rows = []
for b, a in acc.items():
    ts = pd.to_datetime(pd.Series(a["ts"]), errors="coerce").dropna()
    dur = (ts.max() - ts.min()).total_seconds() if len(ts) > 1 else 0.0
    rows.append((b, split.loc[b, "split"], int(split.loc[b, "label"]), a["seq"], a["line_idx"], len(a["seq"]), dur))
sessions = pd.DataFrame(rows, columns=["block_id", "split", "label", "seq", "line_idx", "n_lines", "duration_s"])
sessions.to_pickle(SESSIONS_PKL)
with open(VOCAB_JSON, "w") as f:
    json.dump(sorted(int(k) for k in json.load(open(TEMPLATES_JSON))), f)

print(f"{len(sessions)} sessions; average length {sessions.n_lines.mean():.1f} lines")
print(sessions.groupby(["split", "label"]).agg(n=("block_id", "count"), avg_lines=("n_lines", "mean"), avg_duration_s=("duration_s", "mean")).round(1).to_string())
tm = {int(k): v["template"] for k, v in json.load(open(TEMPLATES_JSON)).items()}
for lab, name in [(0, "NORMAL"), (1, "ANOMALOUS")]:
    s = sessions[(sessions.split == "test") & (sessions.label == lab)].iloc[0]
    print(f"\n--- Example {name} session {s.block_id} ({s.n_lines} lines) ---")
    for j, t in enumerate(s.seq, 1): print(f"  L{j}: {tm[t]}")

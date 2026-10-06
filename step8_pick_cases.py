"""STEP 8: shortlist demo sessions: one agreed anomaly, one agreed normal, one disagreement.
Prints candidates so the team can choose; writes outputs/demo_cases.json (edit block_ids by hand if you prefer others)."""
import json, sys
import pandas as pd
from config import *

if not CONS_CSV.exists(): sys.exit("Run step7_consistency.py first.")
d = pd.read_csv(CONS_CSV)
cands = {
    "agree_anomaly": d[(d.status == "Verified Agree (Anomaly)") & d.valid_citations & (d.label == 1)].sort_values("stat_score", ascending=False),
    "agree_normal": d[(d.status == "Agree (Normal)") & (d.label == 0)].sort_values("stat_score"),
    # prefer a disagreement where the statistical engine raised a false alarm that the LLM correctly dismissed
    "disagreement": d[d.status == "Needs Review"].assign(_p=lambda x: ((x.stat_decision == "Anomalous") & (x.label == 0)).astype(int)).sort_values(["_p", "stat_score"], ascending=False),
}
chosen = []
for role, df in cands.items():
    print(f"\n== {role}: {len(df)} candidates (top 3) ==")
    for r in df.head(3).itertuples():
        print(f"  {r.block_id} | stat {r.stat_score} {r.stat_decision} | LLM {r.llm_decision} | truth {'Anomaly' if r.label else 'Normal'}")
    if len(df): chosen.append({"role": role, "block_id": df.iloc[0].block_id})
    else: print("  (none found - say so in the demo, or adjust sampling/prompt)")
json.dump(chosen, open(CASES_JSON, "w"), indent=1)
print(f"\nWrote {CASES_JSON.name} with {len(chosen)} cases.")

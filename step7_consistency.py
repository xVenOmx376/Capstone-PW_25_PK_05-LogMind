"""STEP 7: consistency layer + metrics.
Both anomalous -> 'Verified Agree (Anomaly)'; both normal -> 'Agree (Normal)'; otherwise 'Needs Review'.
Reports stat-only, LLM-only and verified-only numbers as a BASELINE (no claim that consistency wins - report what you measure).
Expected: if Needs Review is > ~50% of sessions, the LLM prompt probably over-flags: revisit step 5."""
import json, sys
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from config import *

if not STAT_CSV.exists() or not LLM_JSON.exists(): sys.exit("Run steps 4 and 6 first.")
stat = pd.read_csv(STAT_CSV)
llm = json.load(open(LLM_JSON))
stat["llm_decision"] = stat.block_id.map(lambda b: llm.get(b, {}).get("verdict"))
stat["valid_citations"] = stat.block_id.map(lambda b: bool(llm.get(b, {}).get("valid_citations")))
stat = stat[stat.llm_decision.notna()].copy()   # sessions where the LLM produced a usable verdict


def status(r):
    if r.stat_decision == r.llm_decision == "Anomalous": return "Verified Agree (Anomaly)"
    if r.stat_decision == r.llm_decision == "Normal": return "Agree (Normal)"
    return "Needs Review"


stat["status"] = stat.apply(status, axis=1)
stat.to_csv(CONS_CSV, index=False)


def prf(pred):
    p, r, f, _ = precision_recall_fscore_support(stat.label, pred, average="binary", zero_division=0)
    return {"flagged": int(pred.sum()), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


systems = {"Statistical only": prf((stat.stat_decision == "Anomalous").astype(int)),
           "LLM only": prf((stat.llm_decision == "Anomalous").astype(int)),
           "Consistency (verified anomalies only)": prf((stat.status == "Verified Agree (Anomaly)").astype(int))}
rev = stat[stat.status == "Needs Review"]
extra = {"n_sessions": len(stat), "agreement_rate": round(1 - len(rev) / len(stat), 3),
         "needs_review_share": round(len(rev) / len(stat), 3),
         "anomaly_rate_inside_review": round(float(rev.label.mean()), 3) if len(rev) else None,
         "llm_valid_json_rate": round(sum(v["valid_json"] for v in llm.values()) / len(llm), 3),
         "llm_valid_citation_rate": round(sum(v["valid_citations"] for v in llm.values()) / len(llm), 3),
         "llm_modes": sorted({v["mode"] for v in llm.values()})}
json.dump({"systems": systems, "extra": extra}, open(METRICS_JSON, "w"), indent=1)

print(f"{'System':42s} {'flagged':>8s} {'prec':>6s} {'recall':>7s} {'F1':>6s}")
for k, v in systems.items(): print(f"{k:42s} {v['flagged']:>8d} {v['precision']:>6.3f} {v['recall']:>7.3f} {v['f1']:>6.3f}")
print(f"\nSessions scored by both engines: {extra['n_sessions']} | agreement {extra['agreement_rate']:.1%} | "
      f"Needs Review {extra['needs_review_share']:.1%} (of which truly anomalous: {extra['anomaly_rate_inside_review']})")
print(f"LLM valid JSON {extra['llm_valid_json_rate']:.1%} | valid citations {extra['llm_valid_citation_rate']:.1%} | modes {extra['llm_modes']}")
if extra["needs_review_share"] > 0.5: print("WARNING: >50% routed to review - check the LLM prompt (step 5).")
if "STUB" in extra["llm_modes"]: print("NOTE: LLM results come from the STUB placeholder, not a real model.")

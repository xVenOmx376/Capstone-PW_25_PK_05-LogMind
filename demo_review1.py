"""LogMind Review-1 demo (STEP 9). Every number printed here is loaded from computed results.
  python demo_review1.py             cached walk-through of the chosen demo cases
  python demo_review1.py --live      also re-runs the LLM judge live on the first case
  python demo_review1.py --summary   prints the baseline metrics table
"""
import argparse, json, sys, time
import pandas as pd
from config import *
from common import *
from llm_judge import judge

ap = argparse.ArgumentParser()
ap.add_argument("--live", action="store_true"); ap.add_argument("--summary", action="store_true")
ap.add_argument("--fast", action="store_true", help="no pauses"); ap.add_argument("--model", default=LLM_MODEL)
a = ap.parse_args()
pause = (lambda s: None) if a.fast else time.sleep
W = 78
sep = lambda c="-": print(c * W)

if not CASES_JSON.exists(): sys.exit("Run steps 1-8 first (demo_cases.json missing).")
cases = json.load(open(CASES_JSON))
S, tm = load_sessions().set_index("block_id"), load_templates()
parsed = pd.read_csv(PARSED_CSV, dtype={"blocks": str}).fillna({"blocks": ""}).set_index("line_idx")
stat = pd.read_csv(STAT_CSV).set_index("block_id")
cons = pd.read_csv(CONS_CSV).set_index("block_id")
llm = json.load(open(LLM_JSON))
modes = sorted({v["mode"] for v in llm.values()})

sep("="); print("LogMind: Capstone Phase-1 Review-1 prototype demo | Team PW_25_PK_05"); sep("=")
print(f"Data   : {'SYNTHETIC dry-run data (NOT real HDFS)' if SYNTHETIC_FLAG.exists() else 'LogHub HDFS sample'} "
      f"({len(S)} sampled sessions: {int((S.split=='train').sum())} train / {int((S.split=='test').sum())} test)")
print(f"Engines: Isolation Forest (statistical) + LLM judge ({', '.join(modes)}) + consistency layer")
if "STUB" in modes: print("!! LLM = keyword STUB placeholder, not a real model !!")

if a.summary:
    m = json.load(open(METRICS_JSON)); sep()
    print(f"{'System':42s} {'flagged':>8s} {'prec':>6s} {'recall':>7s} {'F1':>6s}")
    for k, v in m["systems"].items(): print(f"{k:42s} {v['flagged']:>8d} {v['precision']:>6.3f} {v['recall']:>7.3f} {v['f1']:>6.3f}")
    e = m["extra"]; print(f"\nAgreement {e['agreement_rate']:.1%} | Needs Review {e['needs_review_share']:.1%} | "
                          f"LLM valid JSON {e['llm_valid_json_rate']:.1%} | valid citations {e['llm_valid_citation_rate']:.1%}")
    print("Baseline on a small sample - not a final result."); sys.exit()

for n, c in enumerate(cases, 1):
    b = c["block_id"]; s = S.loc[b]; st = stat.loc[b]; cr = cons.loc[b]; L = llm[b]
    sep("="); print(f"Scenario {n} [{c['role']}]  session {b}  ({s.n_lines} log lines)"); sep()
    print("[Stage 1] Drain parsing - raw line -> template")
    for li, t in list(zip(s.line_idx, s.seq))[:3]:
        print(f"   RAW : {parsed.loc[li, 'raw'][:100]}\n   TMPL: {tm[t]}")
    pause(0.5)
    print("\n[Stage 2a] Statistical engine (Isolation Forest)")
    print(f"   score {st.stat_score} (threshold {STAT_THRESHOLD}) -> {st.stat_decision}\n   evidence: {st.evidence}")
    pause(0.5)
    print(f"\n[Stage 2b] LLM judge ({L['mode']})")
    print(f"   verdict: {L['verdict']} | explanation: {L['explanation']}")
    for ci in L["cited_lines"]:
        if 1 <= ci <= len(L["shown_lines"]): print(f"   cites L{ci}: {L['shown_lines'][ci-1]}")
    print(f"   citation check: {'PASS' if L['valid_citations'] else 'FAIL'}")
    pause(0.5)
    print("\n[Stage 3] Consistency layer")
    print(f"   Status: [ {cr.status.upper()} ]")
    print("   Action: " + ("emit verified alert" if cr.status.startswith("Verified") else "no alert" if cr.status.startswith("Agree") else "escalate to 'Needs Review' queue for an analyst"))
    print(f"   (ground truth label: {'Anomaly' if cr.label else 'Normal'})")
    if a.live and n == 1:
        print("\n[LIVE] re-running LLM judge now ...")
        t0 = time.time(); r = judge([tm[t] for t in s.seq], a.model)
        print(f"   verdict: {r['verdict']} | {r['explanation']} | citations {'PASS' if r['valid_citations'] else 'FAIL'} ({time.time()-t0:.1f}s)")
    pause(0.8)
sep("="); print("Prototype on a small sample; run `--summary` for baseline metrics. Full-scale evaluation is Phase-II."); sep("=")

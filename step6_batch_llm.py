"""STEP 6: run the LLM judge on all test sessions and cache every output (resumable).
Expected: llm_results.json with ~200 entries. GPU: seconds each; CPU: 10-40 s each -> run overnight if needed."""
import argparse, json, sys, time
from config import *
from common import *
from llm_judge import judge

ap = argparse.ArgumentParser()
ap.add_argument("--stub", action="store_true", help="placeholder keyword judge (dry run only)")
ap.add_argument("--model", default=LLM_MODEL)
ap.add_argument("--limit", type=int, default=0, help="only first N test sessions (smoke test)")
a = ap.parse_args()
if not SESSIONS_PKL.exists(): sys.exit("Run steps 1-3 first.")

S, tm = load_sessions(), load_templates()
test = S[S.split == "test"]
if a.limit: test = test.head(a.limit)
cache = json.load(open(LLM_JSON)) if LLM_JSON.exists() else {}
t0, done = time.time(), 0
for n, s in enumerate(test.itertuples(), 1):
    if s.block_id in cache: continue
    try:
        cache[s.block_id] = judge([tm[t] for t in s.seq], a.model, a.stub)
    except Exception as e:                                   # keep failures in the cache; they are results too
        cache[s.block_id] = {"verdict": None, "valid_json": False, "valid_citations": False, "explanation": "",
                             "cited_lines": [], "shown_lines": [], "mode": "ERROR", "raw": repr(e)}
    done += 1
    if done % 5 == 0 or n == len(test):
        json.dump(cache, open(LLM_JSON, "w"), indent=1)
        print(f"  {n}/{len(test)} done, {(time.time()-t0)/done:.1f}s per session", flush=True)
json.dump(cache, open(LLM_JSON, "w"), indent=1)
vals = [cache[b] for b in test.block_id if b in cache]
print(f"\nLLM results: {len(vals)} sessions | valid JSON {sum(v['valid_json'] for v in vals)/len(vals):.1%} "
      f"| valid citations {sum(v['valid_citations'] for v in vals)/len(vals):.1%} | mode(s): {sorted({v['mode'] for v in vals})}")

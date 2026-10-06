"""STEP 5: LLM Judge (zero-shot) + citation check.
Run `python llm_judge.py` to test the prompt on one normal and one anomalous test session
(add --stub for a keyword placeholder when Ollama is unavailable - for dry runs only)."""
import argparse, json, re
from config import *

SYSTEM = ("You are an expert in Hadoop HDFS log analysis. You judge ONE block session: the ordered log "
          "template lines that mention a single HDFS block. A healthy block is normally allocated, received "
          "by 3 replicas, acknowledged by PacketResponders that terminate, and registered via addStoredBlock. "
          "Missing steps, exceptions, redundant or unexpected operations indicate an anomaly. "
          "Routine background activity (e.g. periodic verification) is NOT an anomaly. Be conservative: "
          "only call a session anomalous when the lines justify it.")

INSTRUCTIONS = ('Return ONLY JSON: {"verdict": "normal" or "anomalous", "explanation": "1-2 sentences", '
                '"cited_lines": [line numbers that support your verdict]}. Cite only line numbers shown below.')


def render_lines(templates, max_lines=MAX_LINES):
    """Number the template lines; if the session is long keep head and tail. Returns shown list of template strings."""
    if len(templates) <= max_lines: return list(templates)
    h = max_lines // 2
    return list(templates[:h]) + list(templates[-(max_lines - h):])


def build_prompt(templates):
    shown = render_lines(templates)
    body = "\n".join(f"L{i}: {t}" for i, t in enumerate(shown, 1))
    note = f"\n(Session has {len(templates)} lines; showing first/last {len(shown)//2} each.)" if len(shown) < len(templates) else ""
    return f"{INSTRUCTIONS}\n\nSession log lines:\n{body}{note}", shown


def parse_llm_output(text, n_shown):
    """Parse model output into a normalized dict and run the citation check."""
    res = {"verdict": None, "explanation": "", "cited_lines": [], "valid_json": False, "valid_citations": False}
    obj = None
    for cand in (text, *re.findall(r"\{.*\}", text or "", flags=re.S)):
        try: obj = json.loads(cand); break
        except Exception: continue
    if not isinstance(obj, dict): return res
    v = str(obj.get("verdict", "")).lower()
    res["verdict"] = "Anomalous" if v.startswith("anom") else "Normal" if v.startswith("norm") else None
    res["valid_json"] = res["verdict"] is not None
    res["explanation"] = str(obj.get("explanation", "")).strip()
    cites = obj.get("cited_lines", [])
    cites = [int(re.sub(r"\D", "", str(c)) or 0) for c in cites] if isinstance(cites, list) else []
    res["cited_lines"] = cites
    in_range = all(1 <= c <= n_shown for c in cites)
    res["valid_citations"] = bool(res["valid_json"] and in_range and (res["verdict"] == "Normal" or len(cites) > 0))
    return res


KEYWORDS = ("exception", "error", "fail", "unexpected", "redundant", "not found", "broken", "reset", "timeout", "invalidset")


def stub_judge(shown):
    hits = [i for i, t in enumerate(shown, 1) if any(k in t.lower() for k in KEYWORDS)]
    if hits: return {"verdict": "anomalous", "explanation": "[STUB] Keyword match on error-like lines.", "cited_lines": hits[:3]}
    return {"verdict": "normal", "explanation": "[STUB] No error-like lines found.", "cited_lines": []}


def judge(templates, model=LLM_MODEL, stub=False):
    prompt, shown = build_prompt(templates)
    if stub:
        text, mode = json.dumps(stub_judge(shown)), "STUB"
    else:
        import ollama
        resp = ollama.chat(model=model, format="json", options={"temperature": 0, "num_ctx": 4096},
                           messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        text, mode = resp["message"]["content"], model
    res = parse_llm_output(text, len(shown))
    res.update(mode=mode, shown_lines=shown, raw=text)
    return res


if __name__ == "__main__":
    from common import load_sessions, load_templates
    ap = argparse.ArgumentParser(); ap.add_argument("--stub", action="store_true"); ap.add_argument("--model", default=LLM_MODEL)
    a = ap.parse_args()
    S, tm = load_sessions(), load_templates()
    test = S[S.split == "test"]
    for lab in (0, 1):
        s = test[test.label == lab].iloc[0]
        r = judge([tm[t] for t in s.seq], a.model, a.stub)
        print(f"\n=== {s.block_id} (ground truth: {'Anomaly' if lab else 'Normal'}) mode={r['mode']} ===")
        print(f"verdict={r['verdict']} json_ok={r['valid_json']} citations_ok={r['valid_citations']}")
        print(f"explanation: {r['explanation']}\ncited: {[(c, r['shown_lines'][c-1]) for c in r['cited_lines'] if 1 <= c <= len(r['shown_lines'])]}")

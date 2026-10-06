"""STEP 2: parse sampled lines with Drain.
Expected: roughly 30-50 templates (LogHub ground truth for HDFS has 29). Hundreds => masking is not working."""
import json, re, sys
import pandas as pd
from datetime import datetime
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction
from config import *

HEADER = re.compile(r"^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([^:\s]+):\s*(.*)$")


def make_miner():
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th, cfg.drain_depth, cfg.profiling_enabled = 0.5, 5, False
    cfg.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+", "BLK"),
        MaskingInstruction(r"\d+\.\d+\.\d+\.\d+(:\d+)?", "IP"),
        MaskingInstruction(r"(?<![\w<])/(?:[\w.\-]+/)+[\w.\-]*", "PATH"),
        MaskingInstruction(r"\b\d+\b", "NUM"),
    ]
    return TemplateMiner(config=cfg)


if not SAMPLE_LOG.exists():
    sys.exit("Run step1_sample_data.py first.")

miner, rows = make_miner(), []
with open(SAMPLE_LOG, errors="replace") as f:
    for i, raw in enumerate(f):
        raw = raw.rstrip("\n")
        m = HEADER.match(raw)
        if m:
            d, t, _pid, level, comp, content = m.groups()
            try: ts = datetime.strptime(d + t, "%y%m%d%H%M%S").isoformat()
            except ValueError: ts = ""
        else:
            ts, level, comp, content = "", "", "", raw
        res = miner.add_log_message(content)
        blocks = "|".join(dict.fromkeys(BLOCK_RE.findall(raw)))
        rows.append((i, ts, level, comp, raw, content, res["cluster_id"], blocks))

# templates evolve while parsing, so read the FINAL template for each cluster
final = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
df = pd.DataFrame(rows, columns=["line_idx", "timestamp", "level", "component", "raw", "content", "template_id", "blocks"])
df.to_csv(PARSED_CSV, index=False)
counts = df.template_id.value_counts()
with open(TEMPLATES_JSON, "w") as f:
    json.dump({str(k): {"template": v, "count": int(counts.get(k, 0))} for k, v in final.items()}, f, indent=1)

print(f"Parsed {len(df)} lines into {len(final)} templates\n")
print("Raw line  ->  template (one example per template, top 8 by frequency)")
for tid in counts.index[:8]:
    ex = df[df.template_id == tid].iloc[0]
    print(f"  RAW : {ex.raw[:110]}\n  TMPL: {final[tid]}\n")
if len(final) > 100:
    print("WARNING: >100 templates - check the masking rules in make_miner().")

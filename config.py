"""Shared settings for the LogMind Review-1 demo. Edit values here, not in the step scripts."""
import re
from pathlib import Path

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

# ---- inputs (download from LogHub: HDFS_v1 -> HDFS.log and preprocessed/anomaly_label.csv)
RAW_LOG = DATA / "HDFS.log"
RAW_LABELS = DATA / "anomaly_label.csv"
SYNTHETIC_FLAG = DATA / ".synthetic"          # created by make_synthetic_data.py

# ---- intermediate files
SAMPLE_LOG = DATA / "sample.log"
SPLIT_CSV = DATA / "sessions_split.csv"
PARSED_CSV = DATA / "parsed.csv"
TEMPLATES_JSON = DATA / "templates.json"
VOCAB_JSON = DATA / "vocab.json"
SESSIONS_PKL = DATA / "sessions.pkl"

# ---- outputs
STAT_CSV = OUT / "stat_results.csv"
STAT_METRICS = OUT / "stat_metrics.json"
LLM_JSON = OUT / "llm_results.json"
CONS_CSV = OUT / "consistency_results.csv"
METRICS_JSON = OUT / "metrics.json"
CASES_JSON = OUT / "demo_cases.json"

# ---- sampling (step 1)
SEED = 42
N_TRAIN = 4000          # normal blocks used to fit the statistical engine
N_TEST_NORMAL = 100
N_TEST_ANOMALY = 100

# ---- statistical engine (step 4)
STAT_THRESHOLD = 0.95   # flag if score >= this (score = percentile among normal training sessions)

# ---- LLM judge (steps 5-6)
LLM_MODEL = "llama3.1:8b"     # use "llama3.2:3b" on weaker machines
MAX_LINES = 40                # max template lines shown to the LLM per session

BLOCK_RE = re.compile(r"blk_-?\d+")




# LogMind - Review-1 Prototype Demo (Team PW_25_PK_05)

Thin end-to-end slice: **HDFS sample -> Drain -> Isolation Forest + LLM judge -> consistency layer**.  
Every number the demo prints is loaded from computed results. This is a small-sample baseline, not a final result.

## Setup (Step 0)


python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Ollama: install from ollama.com, then   ollama pull llama3.1:8b   (or llama3.2:3b on weaker laptops)

Download **HDFS_v1** from LogHub and put `HDFS.log` and `anomaly_label.csv` (in its `preprocessed/` folder) into `data/`.

Change the model, sample sizes, or threshold in `config.py`.

## Run

`./run_all.sh` runs steps 1-8 (resumable LLM step). Or run them one at a time:

| Step | Command | Expected result |
|---|---|---|
| 1 | `python step1_sample_data.py` | 4000 train-normal / 100 test-normal / 100 test-anomaly blocks, no overlap; ~80k-120k lines kept |
| 2 | `python step2_parse_drain.py` | ~30-50 templates (LogHub reference: 29). Hundreds = fix masking in `make_miner()` |
| 3 | `python step3_build_sessions.py` | ~4,200 sessions; prints one normal and one anomalous session in full |
| 4 | `python step4_stat_engine.py` | P/R/F1/AUROC on 200 test sessions; AUROC well above 0.5 (~0.8+ is healthy); evidence text per session |
| 5 | `python llm_judge.py` | Valid JSON verdict + explanation + cited lines on 2 sessions; citation check passes/fails |
| 6 | `python step6_batch_llm.py` | `outputs/llm_results.json` with ~200 entries; valid-JSON and valid-citation rates (resumable, run overnight on CPU) |
| 7 | `python step7_consistency.py` | Metrics table (stat / LLM / verified-only), agreement rate, Needs Review share. Over ~50% review = revisit the prompt |
| 8 | `python step8_pick_cases.py` | `outputs/demo_cases.json`: agreed anomaly, agreed normal, one disagreement (edit block IDs by hand if you like) |
| 9 | `python demo_review1.py` | 3-case walkthrough; `--live` re-runs the LLM on case 1; `--summary` prints metrics; `--fast` skips pauses |

## Dry run (no dataset / no Ollama)

`./run_all.sh --dry-run` generates synthetic HDFS-style data and uses a keyword **STUB** instead of an LLM, to check the pipeline works. The demo banner then says SYNTHETIC / STUB. **Never present dry-run numbers as results.**

## Before you present

- Run `python demo_review1.py --live` once on the demo laptop to measure LLM latency; keep cached mode as the fallback.
- Update slide 8 to show real output from this script (real scores, real block IDs).
- Be ready to say: sample of ~4,200 sessions, baseline only, the LLM prompt is zero-shot, full-scale evaluation is Phase-II.
- If no disagreement exists in your sample, say so honestly instead of staging one.

## Files

`config.py` settings | `common.py` helpers | `llm_judge.py` prompt, JSON parsing, citation check, stub  
`make_synthetic_data.py` dry-run data only | `step*.py` pipeline | `demo_review1.py` the live demo script

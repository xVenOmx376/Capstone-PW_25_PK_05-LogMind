#!/usr/bin/env bash
# Usage:  ./run_all.sh              real data + real Ollama model
#         ./run_all.sh --dry-run    synthetic data + stub LLM (pipeline test only; results are meaningless)
set -e
if [ "$1" == "--dry-run" ]; then
  python make_synthetic_data.py; STUB="--stub"
else
  STUB=""
fi
python step1_sample_data.py
python step2_parse_drain.py
python step3_build_sessions.py
python step4_stat_engine.py
python step6_batch_llm.py $STUB
python step7_consistency.py
python step8_pick_cases.py
echo; echo "Now run:  python demo_review1.py   (or --live / --summary)"

#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python scripts/run_llm_player_evaluation.py \
  --provider mock \
  --hands 24 \
  --seed-start 8100 \
  --seed-count 6 \
  --equity-samples 2 \
  --opponents tag calling_station closed_loop_shaper \
  --output results/llm_player/mock_evaluation
python scripts/analyze_llm_player_evaluation.py

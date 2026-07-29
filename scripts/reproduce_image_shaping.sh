#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=${PYTHONPATH:-src} python -m reflexive_poker image-shaping --output results/image_shaping/static --seeds 80 --hands 320
PYTHONPATH=${PYTHONPATH:-src} python -m reflexive_poker image-shaping --output results/image_shaping/hidden_shift --seeds 40 --hands 320 --hidden-shift

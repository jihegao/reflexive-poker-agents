#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=${PYTHONPATH:-src} python -m reflexive_poker demo --output results/demo

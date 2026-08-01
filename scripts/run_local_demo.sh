#!/usr/bin/env bash
set -euo pipefail

demo_root="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -d "$demo_root/web/node_modules" ]]; then
  npm --prefix "$demo_root/web" install
fi

mkdir -p "$demo_root/.local/poker-demo"
npm --prefix "$demo_root/web" run build

cd "$demo_root"
uv run --extra web reflexive-poker-demo

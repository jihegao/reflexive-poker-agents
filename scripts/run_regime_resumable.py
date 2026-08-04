from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from reflexive_poker.regime_runner import (
    regime_run_config_from_mapping,
    run_regime_experiment_resumable,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume the checkpointed regime-adaptation experiment"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/regime_pilot.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--max-blocks", type=int)
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("config root must be a mapping")
    section = payload.get("regime_adaptation")
    if not isinstance(section, dict):
        raise SystemExit("config requires a regime_adaptation mapping")
    output_dir = args.output or Path(
        section.get("output_dir", "results/regime_adaptation/formal_pilot_v1")
    )
    config = regime_run_config_from_mapping(
        payload,
        output_dir=output_dir,
        run_id=args.run_id,
        max_blocks=args.max_blocks,
    )
    status = run_regime_experiment_resumable(config)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

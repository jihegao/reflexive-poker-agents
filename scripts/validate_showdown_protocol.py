from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexive_poker.showdown_protocol import (
    load_showdown_protocol,
    protocol_fingerprint,
    validate_showdown_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the DeepSeek-vs-Luna poker protocol")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deepseek_v4_flash_vs_gpt_5_6_luna.yaml"),
    )
    parser.add_argument("--formal", action="store_true", help="Require a fully frozen formal plan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_showdown_protocol(args.config)
    errors = validate_showdown_protocol(payload, formal=args.formal)
    result = {
        "valid": not errors,
        "mode": "formal" if args.formal else "protocol",
        "config": str(args.config),
        "sha256": protocol_fingerprint(payload),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

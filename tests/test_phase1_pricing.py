from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from reflexive_poker.phase1_pricing import PricingManifestError, resolve_phase1_pricing


def _config_and_manifest(tmp_path: Path) -> tuple[Path, dict]:
    pricing = {
        "schema_version": 1,
        "protocol": "prbench-cross-model-v1",
        "frozen": True,
        "frozen_at_utc": "2026-08-02T05:00:00+00:00",
        "entries": [
            {"provider": "opencode-go", "model": "deepseek-v4-flash", "cost_observability": "exact"},
            {"provider": "codex", "model": "gpt-5.6-luna", "cost_observability": "unavailable", "unavailable_reason": "CLI emits no bill"},
        ],
    }
    price_path = tmp_path / "pricing.json"
    price_path.write_text(json.dumps(pricing), encoding="utf-8")
    config = {
        "protocol": "prbench-cross-model-v1",
        "paper_phase1": {
            "models": [
                {"provider": "opencode-go", "model": "deepseek-v4-flash"},
                {"provider": "codex", "model": "gpt-5.6-luna"},
            ],
            "pricing_manifest": {
                "path": "pricing.json",
                "sha256": hashlib.sha256(price_path.read_bytes()).hexdigest(),
            },
        },
    }
    config_path = tmp_path / "phase1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def test_phase1_pricing_is_hash_locked_and_predates_the_run(tmp_path: Path) -> None:
    config_path, config = _config_and_manifest(tmp_path)
    frozen = resolve_phase1_pricing(
        config_path,
        config,
        run_created_at="2026-08-02T05:01:00+00:00",
    )
    assert frozen is not None
    assert frozen.sha256 == config["paper_phase1"]["pricing_manifest"]["sha256"]


def test_phase1_pricing_rejects_tampering_missing_system_and_postdated_snapshot(tmp_path: Path) -> None:
    config_path, config = _config_and_manifest(tmp_path)
    price_path = tmp_path / "pricing.json"
    price_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PricingManifestError, match="SHA-256"):
        resolve_phase1_pricing(config_path, config)

    config_path, config = _config_and_manifest(tmp_path)
    pricing = json.loads((tmp_path / "pricing.json").read_text())
    pricing["entries"] = pricing["entries"][:1]
    (tmp_path / "pricing.json").write_text(json.dumps(pricing), encoding="utf-8")
    config["paper_phase1"]["pricing_manifest"]["sha256"] = hashlib.sha256(
        (tmp_path / "pricing.json").read_bytes()
    ).hexdigest()
    with pytest.raises(PricingManifestError, match="cover exactly"):
        resolve_phase1_pricing(config_path, config)

    config_path, config = _config_and_manifest(tmp_path)
    with pytest.raises(PricingManifestError, match="after"):
        resolve_phase1_pricing(
            config_path, config, run_created_at="2026-08-02T04:59:00+00:00"
        )

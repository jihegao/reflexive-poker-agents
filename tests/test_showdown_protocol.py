from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from reflexive_poker.showdown_protocol import (
    REQUIRED_ARTIFACTS,
    load_showdown_protocol,
    protocol_fingerprint,
    validate_showdown_protocol,
)

CONFIG = Path("configs/deepseek_v4_flash_vs_gpt_5_6_luna.yaml")


def test_checked_in_protocol_contract_is_valid() -> None:
    payload = load_showdown_protocol(CONFIG)

    assert validate_showdown_protocol(payload) == []
    assert REQUIRED_ARTIFACTS.issubset(set(payload["required_artifacts"]))
    assert len(protocol_fingerprint(payload)) == 64


def test_formal_mode_fails_until_gto_pack_and_protocol_are_frozen() -> None:
    payload = load_showdown_protocol(CONFIG)

    errors = validate_showdown_protocol(payload, formal=True)

    assert any("protocol.status" in error for error in errors)
    assert any("GTO reference pack" in error for error in errors)


def test_provider_fallbacks_are_fail_closed() -> None:
    payload = deepcopy(load_showdown_protocol(CONFIG))
    payload["provider_gate"]["require_zero_fallbacks"] = False
    payload["provider_gate"]["rule_bot_takeover_allowed"] = True

    errors = validate_showdown_protocol(payload)

    assert any("require_zero_fallbacks" in error for error in errors)
    assert any("rule_bot_takeover_allowed" in error for error in errors)

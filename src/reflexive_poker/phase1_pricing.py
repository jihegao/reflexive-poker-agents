"""Immutable price-snapshot validation for formal Phase 1 evidence.

The snapshot is a *frozen input*, not a best-effort post-hoc cost lookup.  It
therefore has to be hashed in the preregistration config before a worker is
allowed to start.  Per-call provider bills remain the primary cost evidence;
the rate schedule only makes API-equivalent estimates reproducible when that
bill is unavailable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class PricingManifestError(ValueError):
    """The formal run cannot be started with the supplied price snapshot."""


@dataclass(frozen=True)
class FrozenPricingManifest:
    path: Path
    payload: dict[str, Any]
    sha256: str
    frozen_at_utc: str


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise PricingManifestError(f"{field} must be a non-empty ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PricingManifestError(f"{field} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise PricingManifestError(f"{field} must include a UTC offset")
    return parsed


def _phase_models(config: dict[str, Any]) -> set[tuple[str, str]]:
    phase = config.get("paper_phase1")
    if not isinstance(phase, dict):
        return set()
    models = phase.get("models")
    if not isinstance(models, list):
        raise PricingManifestError("paper_phase1.models must be a list")
    pairs: set[tuple[str, str]] = set()
    for item in models:
        if not isinstance(item, dict) or not item.get("provider") or not item.get("model"):
            raise PricingManifestError("each Phase 1 pricing model needs provider and model")
        pairs.add((str(item["provider"]), str(item["model"])))
    return pairs


def resolve_phase1_pricing(
    config_path: Path,
    config: dict[str, Any],
    *,
    run_created_at: str | None = None,
) -> FrozenPricingManifest | None:
    """Validate the frozen Phase 1 price input and return its exact bytes' hash.

    A config without ``paper_phase1`` is Phase 2-only and does not need this
    input.  A Phase 1 config, by contrast, must never silently proceed without
    it, including for baseline/preflight runs that become part of the package.
    """
    phase = config.get("paper_phase1")
    if phase is None:
        return None
    if not isinstance(phase, dict):
        raise PricingManifestError("paper_phase1 must be a mapping")
    reference = phase.get("pricing_manifest")
    if not isinstance(reference, dict):
        raise PricingManifestError("paper_phase1.pricing_manifest must lock path and sha256")
    relative_path = reference.get("path")
    expected_sha256 = reference.get("sha256")
    if not isinstance(relative_path, str) or not relative_path:
        raise PricingManifestError("pricing_manifest.path must be a non-empty relative path")
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise PricingManifestError("pricing_manifest.path must stay inside the repository config tree")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise PricingManifestError("pricing_manifest.sha256 must be a SHA-256 digest")
    path = (config_path.parent / relative_path).resolve()
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise PricingManifestError(f"pricing manifest does not exist: {path}") from exc
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PricingManifestError("pricing manifest SHA-256 does not match the preregistration lock")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PricingManifestError("pricing manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PricingManifestError("pricing manifest root must be an object")
    if payload.get("schema_version") != 1:
        raise PricingManifestError("pricing manifest schema_version must be 1")
    if payload.get("protocol") != config.get("protocol"):
        raise PricingManifestError("pricing manifest protocol does not match the preregistration")
    if payload.get("frozen") is not True:
        raise PricingManifestError("pricing manifest must be explicitly frozen")
    frozen_at = _parse_utc(payload.get("frozen_at_utc"), field="frozen_at_utc")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise PricingManifestError("pricing manifest entries must be a list")
    observed: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PricingManifestError("pricing manifest entries must be objects")
        provider, model = entry.get("provider"), entry.get("model")
        if not isinstance(provider, str) or not isinstance(model, str):
            raise PricingManifestError("each pricing manifest entry needs provider and model")
        mode = entry.get("cost_observability")
        if mode not in {"exact", "estimated", "unavailable"}:
            raise PricingManifestError("cost_observability must be exact, estimated, or unavailable")
        if mode == "unavailable" and not entry.get("unavailable_reason"):
            raise PricingManifestError("unavailable pricing must state why it is unavailable")
        if mode == "estimated" and not isinstance(entry.get("rates_usd_per_million"), dict):
            raise PricingManifestError("estimated pricing must include a frozen rate schedule")
        observed.add((provider, model))
    required = _phase_models(config)
    if observed != required:
        raise PricingManifestError(
            "pricing manifest must cover exactly the frozen Phase 1 provider/model pairs"
        )
    if run_created_at is not None and frozen_at > _parse_utc(
        run_created_at, field="run.created_at"
    ):
        raise PricingManifestError("pricing manifest was frozen after this formal run was created")
    return FrozenPricingManifest(
        path=path,
        payload=payload,
        sha256=actual_sha256,
        frozen_at_utc=str(payload["frozen_at_utc"]),
    )

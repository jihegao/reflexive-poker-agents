# v0.5.0 - Auditable LLMPlayer

This release adds a provider-backed poker agent with structured per-decision reasoning summaries and post-hand reflections.

## Added

- `LLMPlayer` and `LLMProvider` protocol.
- OpenAI Responses API provider with strict JSON Schema Structured Outputs.
- Deterministic offline mock provider for CI, replay, and integration evaluation.
- Legal-action validation, schema validation, and safe fallback to the existing rule policy.
- Bounded reflection memory fed into subsequent decision prompts.
- Gzip JSONL decision and reflection traces.
- Seat-mirrored evaluation against TAG, Calling Station, and Closed-loop Shaper.
- Offline trace viewer, metrics, charts, and a Chinese technical report.

## Frozen offline integration run

- 6 paired seeds per opponent.
- 24 hands per seat mirror.
- 2,561 decision traces.
- 864 post-hand reflections.
- 100% legal final actions.
- 0 fallbacks, invalid actions, or provider failures.
- 100% required audit-field completeness.

## Evidence boundary

The frozen evaluation uses `DeterministicNarrativeProvider`, not a live LLM, because the execution environment did not contain `OPENAI_API_KEY`. The run validates implementation and traceability only. Profit values must not be presented as GPT-5 mini or real-model performance.

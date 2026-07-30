# Reflexive Poker Agents

A reproducible multi-agent poker testbed for situated self-models, strategic public-image shaping, player-type matchups, and auditable LLM-driven decisions.

## v0.5.0: LLMPlayer

Version 0.5 adds a provider-backed `LLMPlayer` that requests one structured action at every decision point and one structured reflection after every completed hand.

Implemented features:

- OpenAI Responses API adapter with strict JSON Schema Structured Outputs;
- deterministic offline provider for CI, replay, and integration testing;
- legal-action validation and safe fallback to the existing rule policy;
- bounded reflection memory passed into later decisions;
- per-decision audit fields: confidence, concise rationale, self-model, opponent model, risk flags, and next-step plan;
- post-hand reflection fields: outcome review, belief update, strategy adjustment, and confidence calibration;
- provider/model identity, latency, token usage, response ID, and error logging;
- seat-mirrored evaluation against TAG, Calling Station, and Closed-loop Shaper.

The checked-in mock evaluation validates the engineering contract only. It is not evidence about GPT-5 mini or any other live model because the frozen run used `DeterministicNarrativeProvider` without an API key.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Install the optional OpenAI SDK integration:

```bash
python -m pip install -e '.[llm]'
```

## Run the offline integration evaluation

```bash
./scripts/reproduce_llm_player.sh
```

This produces seat-mirrored match summaries plus gzip JSONL decision and reflection traces under `results/llm_player/`.

## Run a live OpenAI smoke test

```bash
export OPENAI_API_KEY='...'
PYTHONPATH=src python scripts/run_llm_player_evaluation.py \
  --provider openai \
  --model gpt-5-mini \
  --hands 8 \
  --seed-count 2 \
  --opponents tag calling_station \
  --output results/llm_player/openai_smoke
```

Start small: a full replacement of the frozen mock run would require thousands of model calls.

## Earlier experiments

The repository also contains the situated-reflection, image-shaping, and player-type matchup environments used in earlier project phases. These experiments consistently separate mechanism claims from poker-profit claims: improved self-model fidelity or belief control does not by itself establish higher reward.

## Evidence boundary

The system stores model-provided concise audit summaries, not hidden chain-of-thought. The poker environment is a transparent research simulator with discretized betting and heuristic opponents; it is not a competitive poker solver.

See `docs/llm_player.md`, `docs/release_v0.5.0.md`, and `configs/llm_player.yaml`.

## Citation and license

See `CITATION.cff`. Licensed under MIT.

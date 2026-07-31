# LLMPlayer

`LLMPlayer` lets the simulator delegate every legal poker action and every post-hand reflection to a provider implementing the `LLMProvider` protocol.

## Providers

### OpenAI Responses API

`OpenAIResponsesProvider` uses the Responses API with strict JSON Schema output. It requests concise audit fields rather than hidden chain-of-thought:

- action and raise scale;
- confidence;
- situation summary and short rationale;
- self-model and opponent model;
- risk flags and next-step plan;
- post-hand outcome review, belief updates, strategy adjustment, and calibration.

```bash
python -m pip install -e '.[llm]'
export OPENAI_API_KEY='...'
PYTHONPATH=src python scripts/run_llm_player_evaluation.py \
  --provider openai \
  --model gpt-5-mini \
  --hands 8 \
  --seed-count 2 \
  --opponents tag calling_station \
  --output results/llm_player/openai_smoke
```

### OpenCode Go

`OpenCodeGoProvider` invokes the locally authenticated `opencode` CLI, so its `opencode-go` key remains in OpenCode's local credential store and is never written to this repository. It asks `deepseek-v4-flash` for a JSON object matching the same audited decision/reflection schemas.

```bash
PYTHONPATH=src python scripts/run_llm_player_evaluation.py \
  --provider opencode-go \
  --model deepseek-v4-flash \
  --hands 2 \
  --seed-count 1 \
  --opponents tag \
  --output results/llm_player/opencode_go_deepseek_v4_flash_smoke
```

### Local Codex account

`CodexProvider` uses `codex exec --ephemeral --sandbox read-only --json` and the local Codex login. Every simulator decision or reflection starts an independent Codex thread and requests a JSON-Schema-constrained final response. The traces record the token usage reported by each completed Codex turn.

```bash
PYTHONPATH=src python scripts/run_llm_player_evaluation.py \
  --provider codex \
  --model current \
  --hands 1 \
  --seed-count 1 \
  --opponents tag \
  --output results/llm_player/codex_current_smoke
```

This provider is appropriate for small behavioral checks, not large tournaments: each call creates a separate Codex thread and therefore has substantial setup-token and latency overhead.

### Deterministic mock provider

`DeterministicNarrativeProvider` implements the same structured contract without calling a model. It exists for CI, replay, schema validation, logging checks, and offline demonstrations. Its poker results must not be described as LLM performance.

```bash
PYTHONPATH=src python scripts/run_llm_player_evaluation.py \
  --provider mock \
  --output results/llm_player/mock_evaluation
PYTHONPATH=src python scripts/analyze_llm_player_evaluation.py
```

## Trace files

- `decision_traces.jsonl.gz`: one entry for every decision point;
- `reflection_traces.jsonl.gz`: one entry after every completed hand;
- `trace_viewer.html`: offline interactive trace browser;
- `trace_examples.md`: compact readable samples;
- `matches.csv` and `summary.csv`: evaluation outcomes.

Every decision trace includes the full structured simulator state, legal action list, provider output, final action, fallback status, provider/model identity, latency, token usage, and response ID. Reflections use public actions and the focal player's own cards; they do not expose unseen opponent hole cards.

## Evidence boundary

The current checked-in result uses the mock provider because the execution environment had no `OPENAI_API_KEY`. It validates the end-to-end engineering contract but cannot support claims about GPT-5 mini or any other real model.

# Reflexive Poker Agents

## Local playable table

Run `./scripts/run_local_demo.sh`, then open `http://127.0.0.1:8790`. The demo
supports per-seat Human/Rule/LLM control, per-seat OpenCode Go model selection,
independent persona-based post-hand strategy versions, SQLite recovery, and WebSocket
events. See [docs/local_demo.md](docs/local_demo.md) for the frozen behavior and
verification commands.

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

## Run a live OpenCode Go smoke test

With a locally authenticated `opencode` CLI, the integration reuses its `opencode-go` credential without storing a key in this repository:

```bash
PYTHONPATH=src python scripts/run_llm_player_evaluation.py \
  --provider opencode-go \
  --model deepseek-v4-flash \
  --hands 2 \
  --seed-count 1 \
  --opponents tag \
  --output results/llm_player/opencode_go_deepseek_v4_flash_smoke
```

## Run a local Codex-account smoke test

```bash
PYTHONPATH=src python scripts/run_llm_player_evaluation.py \
  --provider codex \
  --model current \
  --hands 1 \
  --seed-count 1 \
  --opponents tag \
  --output results/llm_player/codex_current_smoke
```

This uses the local Codex login and emits per-turn token accounting in the traces.

## Run the paired six-max second-order experiment

The six-max cash-game experiment pairs identical deck seeds, seats, 100 BB resets,
and five heuristic opponents. `reflexive_off` receives only first-order poker state;
`reflexive_on` additionally receives public self-image, opponent summaries, collective
fold probability, and prior reflections.

```bash
PYTHONPATH=src python scripts/run_second_order_experiment.py \
  --seed-start 9200 \
  --seed-count 1 \
  --hands 6 \
  --equity-samples 64 \
  --workers 1 \
  --models opencode-go:deepseek-v4-flash codex:gpt-5.6-luna \
  --output results/second_order/low_cost_pilot
```

Use `--workers 1` for the clean provider gate. A one-seed, six-hand run validates
the intervention and accounting only; it cannot establish a profitability advantage.
The checked-in clean pilot report is under
`results/second_order/low_cost_clean_pilot_seed9200/`.

## Earlier experiments

The repository also contains the situated-reflection, image-shaping, and player-type matchup environments used in earlier project phases. These experiments consistently separate mechanism claims from poker-profit claims: improved self-model fidelity or belief control does not by itself establish higher reward.

## Phase 1 opponent-modeling experiment

The two-stage PRBench protocol (paper-minimum Phase 1, four-model Phase 2) is documented in
[`docs/prbench_cross_model_experiment_plan.zh-CN.md`](docs/prbench_cross_model_experiment_plan.zh-CN.md).

The Phase 1 runner separates history use, action prediction, strategy-type inference,
the budget-matched non-recursive D1 control, and auditable recursive levels D2/D3. It
uses a shared formation checkpoint, paired
deck/seat seeds, bounded structured belief states, provider-call budgets, large-pot
sensitivity checks, and a fail-closed provider gate. Existing `reflexive_on/off`
artifacts are not relabeled as Phase 1 depth evidence.

Run a short deterministic rule-agent smoke:

```bash
PYTHONPATH=src uv run python scripts/run_phase1_experiment.py \
  --arena heads_up \
  --opponent tag \
  --hands 12 \
  --formation-hands 4 \
  --equity-samples 1 \
  --output results/phase1/rule_smoke
```

Run the complete condition-matrix wiring smoke with one seed and short branches:

```bash
PYTHONPATH=src uv run python scripts/run_phase1_experiment.py \
  --matrix-smoke \
  --output results/phase1/matrix_smoke
```

Real-provider runs are intentionally opt-in. Validate a small zero-failure smoke before
using `--preregistered`; the checked-in defaults and 10,000-call ceilings are documented
in `configs/phase1.yaml`. The runner never silently substitutes a provider or model.
Formal Phase 1 starts also hash-lock and copy the dated
[`price snapshot`](configs/pricing/phase1-2026-08-02.json) into each run before the
worker starts; a missing, changed, incomplete, or post-dated snapshot is fail-closed.
Provider-reported billed cost remains authoritative, while the Codex CLI's unavailable
per-call bill is recorded as unavailable rather than zero.

The frozen four-system Phase 2 preparation manifest is
[`configs/phase2.yaml`](configs/phase2.yaml). It records the intended treatments and
Six-max external-validity contract. `paper-phase2-preflight` can run its bounded four-case,
all-treatment provider gate for the four frozen systems; it does not run any Phase 2 outcome.

Generate the frozen offline cases plus Oracle, Uniform, Frequency, Bayesian, and HMM
controls without model calls:

```bash
uv run expctl run start \
  --config configs/phase1.yaml \
  --experiment offline-baselines \
  --request-id phase1-offline-baselines-v1 \
  --output json
```

### Agent-friendly background control

`expctl` is the stable non-interactive interface for research execution. It launches an
independent low-priority worker, writes only to the experiment registry, emits JSON or
JSONL, and does not use the frontend service or product database.

```bash
uv run expctl doctor --output json
uv run expctl experiment list --output json
uv run expctl config validate configs/phase1.yaml --output json

uv run expctl phase2-readiness \
  --config configs/phase2.yaml \
  --preflight-dir results/experiments/<phase2-preflight-run>/artifacts/preflight \
  --output-dir results/phase2_readiness/current \
  --output json

uv run expctl run start \
  --config configs/phase2.yaml \
  --experiment paper-phase2-preflight \
  --request-id phase2-provider-preflight-v1 \
  --tag phase2-provider-preflight \
  --output json

uv run expctl run start \
  --config configs/phase1.yaml \
  --experiment paper-phase1 \
  --request-id paper-phase1-formal-v1 \
  --tag paper-phase1 \
  --output json

uv run expctl run start \
  --config configs/regime_pilot.yaml \
  --experiment regime-adaptation \
  --request-id regime-formal-pilot-v1 \
  --tag regime-formal-pilot \
  --output json

uv run expctl run status <run-id> --output json
uv run expctl run logs <run-id> --follow --format jsonl
uv run expctl analyze <run-id> --output json
uv run expctl export <run-id> --format tar.gz --output json
```

`run pause`, `run resume`, and `run stop` expose the same machine-readable state
contract. Reusing a `--request-id` is idempotent; using it with a different config or
experiment fails with `IDEMPOTENCY_CONFLICT`.

### Resumable formal runs

The formal executors checkpoint rule experiments per cell/seed and LLM experiments per
model/job/paired-seed block. Completion is atomic. A provider ledger is written before
and after every call, so an interrupted call is conservatively counted against the
budget. Re-run the exact same command after `Ctrl-C`, process termination, or restart;
completed blocks are verified and skipped. A changed plan or model configuration fails
closed instead of mixing protocols.

Formal execution requires a clean Git worktree. The plan freezes the Git commit and a
content fingerprint of the experiment implementation. `--allow-dirty-worktree` exists
only for bounded rehearsal runs; it is recorded in provenance and should not be used for
confirmatory evidence.

Live offline understanding runs additionally append each validated model prediction to
`live_predictions.jsonl` and record the one active call in `LIVE_INFLIGHT.json`.
They resume only when that journal and the provider ledger reconcile exactly; an
interruption with an unrecorded in-flight response fails closed rather than replaying a
call and silently mixing provider accounting.

Start or resume the complete 40-cell, 60-seed rule matrix:

```bash
PYTHONPATH=src uv run python scripts/run_phase1_resumable.py simulation \
  --output results/phase1/full_simulation
```

Start or resume the preregistered dual-model Heads-up confirmation:

```bash
PYTHONPATH=src uv run python scripts/run_phase1_resumable.py llm \
  --selected-depth recursive_d2 \
  --output results/phase1/llm_confirmation
```

Inspect progress without running new blocks:

```bash
PYTHONPATH=src uv run python scripts/run_phase1_resumable.py status \
  --output results/phase1/full_simulation
```

Use `--max-seed-blocks`, `--max-cells`, or `--max-blocks` to bound one invocation;
these controls do not change the frozen experimental plan. Incomplete `.running`
directories are retained under an `interrupted/` directory and are never treated as
evidence. LLM primary-call allocation is checked against a conservative per-block upper
bound before a new block starts; unresolved failures, fallback, identity mismatch,
unbalanced paired calls, or incomplete accounting invalidate that block.

The regime-adaptation pilot uses the same `expctl` lifecycle and adds atomic checkpoints
for every `condition × seed × mirror` block. Its frozen 30-seed config, standalone resume
command, completion artifacts, Student-t paired inference, and fail-closed claim boundary
are documented in
[`docs/regime_adaptation_experiment.md`](docs/regime_adaptation_experiment.md). Long
runtime is handled by checkpoint/resume rather than an unlimited foreground timeout.

## Evidence boundary

The system stores model-provided concise audit summaries, not hidden chain-of-thought. The poker environment is a transparent research simulator with discretized betting and heuristic opponents; it is not a competitive poker solver.

See `docs/llm_player.md`, `docs/release_v0.5.0.md`, and `configs/llm_player.yaml`.

## Citation and license

See `CITATION.cff`. Licensed under MIT.

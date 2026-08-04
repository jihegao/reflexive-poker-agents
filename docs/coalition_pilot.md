# Coalition reflection + simulation pilot

This track tests whether two agents can coordinate from public table actions in
a closed Hold'em simulation. It is separate from the Phase 1/Phase 2 PRBench
protocol and cannot unlock or substitute for either phase.

## Current implementation

`src/reflexive_poker/coalition_experiment.py` runs a paired `2x2x2` mechanism
smoke with two coalition seats and four independent rule-based controls. The
three factors are:

1. team reward (`u_A + lambda * u_B` proxy in the action scorer),
2. public-action reflection, and
3. bounded joint simulation.

Each seed is run across two seat mirrors. The runner writes per-seed and
per-hand rows, a condition summary, a JSON audit, and a short design note.

## Information boundary

The coalition agents receive their own cards and public action events only.
They do not receive the partner's hole cards, and the smoke audit records zero
private-information accesses. The current joint simulation is a deterministic
mechanism proxy; it is not a GTO solver or an LLM provider evaluation.

## Metrics and gates

Primary exploratory metrics are paired coalition surplus relative to `t0r0s0`
and the reflection-by-simulation interaction within the team-reward condition.
Secondary metrics include partner-action dependency, coordination actions,
simulation/reflection counts, control rewards, and the private-information
audit. Smoke artifacts set `formal_conclusion_allowed=false` by design.

Before any live-provider or formal payoff run, add more paired seeds, button and
seat mirroring, a calibrated power analysis, partner replacement and memory
reset controls, and an external counterfactual value evaluator. No result from
this smoke can establish a general profitability or collusion claim.

Run the local smoke with:

```bash
uv run python scripts/run_coalition_experiment.py \
  --seed 9400 --hands 12 --output results/coalition/mock_smoke_v3
```

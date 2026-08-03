# Regime-adaptation experiment

This experiment tests a narrower claim than “LLMs play poker better.” It asks whether a
reflection layer that generates alternative opponent worlds, followed by a bounded
counterfactual simulator, recovers faster after an opponent policy switch.

## Conditions

- `baseline`: the existing rule policy and slow Bayesian-style aggression belief.
- `reflection`: exponentially decayed action-frequency tracking without change-point reset.
- `reflection_simulation`: surprise detection, bounded hypothesis generation, posterior
  scoring, and simulated response selection.

The default CI path uses `HeuristicHypothesisGenerator`. It validates the mechanism and
must not be presented as evidence about a live language model. `ProviderHypothesisGenerator`
uses the repository `LLMProvider.structured(...)` contract when a live-provider study is
explicitly configured.

## Intervention

The paired heads-up opponent uses a frozen TAG policy before `switch_hand` and a frozen
LAG policy afterward. Each seed is seat-mirrored. The environment, deck seed, stack,
blinds, and raise cap are shared across conditions.

## Metrics

- total reward in big blinds;
- post-switch BB/100;
- recovery hands: first point with three consecutive positive trailing windows;
- first post-switch detected change and detection delay;
- hypothesis-generator and simulator call counts.

## Scope limitation

`WorldSimulator` is a decision-layer Monte Carlo model over opponent action frequencies
and three response families (`pressure`, `balanced`, `bluff_catch`). It is not a full
Hold'em rollout engine and does not estimate exploitability. A positive result justifies
replacing this compact simulator with the repository's stronger poker simulation stack;
it does not itself establish equilibrium improvement.

## Smoke run

```bash
PYTHONPATH=src python scripts/run_regime_switch_experiment.py \
  --seed-count 1 \
  --hands 40 \
  --switch-hand 20 \
  --equity-samples 1 \
  --recovery-window 6 \
  --output results/regime_adaptation/smoke
```

## Confirmatory interpretation

The central comparison is `reflection_simulation` versus `reflection`, not versus the
raw baseline. The mechanism is supported only if the enhanced condition has shorter
recovery, non-inferior static-period reward, and bounded additional provider/simulation
cost across preregistered paired seeds.

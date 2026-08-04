# Regime-adaptation experiment

This experiment tests a narrower claim than “LLMs play poker better.” It asks whether a
reflection layer that generates alternative opponent worlds, followed by counterfactual
full-hand simulation, recovers faster after an opponent policy switch.

## Conditions

- `baseline`: the existing rule policy and slow aggression belief.
- `reflection`: exponentially decayed action-frequency tracking without a change-point reset.
- `reflection_simulation`: calibrated surprise detection, bounded hypothesis generation,
  posterior scoring, and full-hand response-policy rollout.

The default offline path uses `HeuristicHypothesisGenerator`. It validates the mechanism
and must not be presented as evidence about a live language model.
`ProviderHypothesisGenerator` uses the repository `LLMProvider.structured(...)` contract
when a live-provider study is explicitly configured.

## Conditional opponent worlds

The previous prototype counted raw `fold`, `check_call`, and `raise` actions. That
confounded checks with calls and treated actions without their legal context. The revised
world model estimates three public, conditional quantities:

- opening raise probability when no bet is faced;
- fold probability when facing a bet;
- reraise probability conditional on continuing versus a bet.

The detector forms an empirical world, observes a separate calibration period, and raises
its threshold from the measured stable-regime surprise distribution before change
detection is enabled. This prevents low fixed thresholds from converting ordinary
formation noise into false strategy switches.

## Full-hand world simulation

`WorldSimulator` now runs the repository `HoldemEnvironment` for every candidate world and
response family. Each rollout includes actual card dealing, blinds, four betting streets,
folds, stack contributions, side pots, and showdown. The three response families are:

- `pressure`;
- `balanced`;
- `bluff_catch`.

For a candidate world, all response families receive the same deck and seat seeds. This
paired common-random-number design reduces variance relative to independent simulations.
The simulator remains a compact policy evaluator rather than an equilibrium solver.

## Intervention

The paired heads-up opponent uses a frozen TAG policy before `switch_hand` and a frozen
LAG policy afterward. Each seed is seat-mirrored. Environment seed, stack, blinds, and
raise cap are shared across conditions.

## Outputs and metrics

The runner writes:

- `matches.csv`: match-level reward, recovery, detection, selected policy, and simulation
  cost;
- `summary.json`: condition-level means, detection rate, and response-policy counts;
- `paired_effects.csv`: seed-and-seat matched `reflection_simulation - reflection`
  differences;
- `paired_summary.json`: paired means with Student-t 95% confidence intervals.

Primary metrics are post-switch BB/100 and recovery hands. Detection delay and simulated
hands quantify mechanism behavior and additional cost.

## Smoke run

```bash
PYTHONPATH=src python scripts/run_regime_switch_experiment.py \
  --seed-count 1 \
  --hands 72 \
  --switch-hand 36 \
  --equity-samples 2 \
  --simulation-rollout-hands 4 \
  --formation-observations 8 \
  --calibration-observations 4 \
  --recovery-window 8 \
  --output results/regime_adaptation/smoke
```

## Bounded development pilot

A five-seed, seat-mirrored development run with 160 hands, a switch at hand 80, four
live-equity samples, and 16 full-hand rollouts per response/world produced:

- post-switch detection in 8 of 10 matches;
- mean detection delay of 11.6 hands among detected matches;
- mean paired post-switch difference of approximately `+4.2 BB/100` versus reflection;
- a very wide 95% interval spanning roughly `-222` to `+230 BB/100`.

This is a wiring and calibration result, not evidence of a profitability advantage. The
central claim remains unsupported until a larger preregistered paired run passes the
repository's provenance, completion, cost, and inference gates.

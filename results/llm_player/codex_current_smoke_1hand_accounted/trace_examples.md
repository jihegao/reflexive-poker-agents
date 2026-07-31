# LLMPlayer trace examples

## Hand 0 · preflop · vs tag

- Cards: `6s Jd`; board: `None`
- Pot / to call: 3.24 / 1.2400000000000002
- Equity / pot odds: 0.0 / 0.2767857142857143
- Action: **fold**; confidence: 0.99
- Situation: Heads-up preflop with zero supplied equity against a call requiring 27.7% pot equity.
- Rationale: Fold: supplied equity is below pot odds, and opponent fold mean offers no fold-equity basis for a raise.
- Self model: Neutral self-image; no prior evidence changes the threshold.
- Opponent model: Aggressive opponent with zero observed fold tendency.
- Risk flags: ['equity_below_pot_odds', 'no_fold_equity']

## Hand 0 · preflop · vs tag

- Cards: `Qd Ah`; board: `None`
- Pot / to call: 1.5 / 0.5
- Equity / pot odds: 1.0 / 0.25
- Action: **raise**; confidence: 0.99
- Situation: Heads-up preflop with 100% supplied equity and 25% pot odds.
- Rationale: Equity overwhelmingly exceeds the calling threshold; raise for value.
- Self model: Neutral prior; no conflicting reflection signal.
- Opponent model: Opponent is modeled as moderately aggressive with average fold tendency.
- Risk flags: []

# Post-hand reflection examples

## Hand 0 · vs tag

- Outcome: None
- Review: None
- Worked: None
- Failed: None
- Belief updates: None
- Adjustment: None
- Calibration: None

## Hand 0 · vs tag

- Outcome: None
- Review: None
- Worked: None
- Failed: None
- Belief updates: None
- Adjustment: None
- Calibration: None

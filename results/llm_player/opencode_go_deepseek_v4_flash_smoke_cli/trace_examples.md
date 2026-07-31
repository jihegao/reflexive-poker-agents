# LLMPlayer trace examples

## Hand 0 · preflop · vs tag

- Cards: `6s Jd`; board: `None`
- Pot / to call: 3.24 / 1.2400000000000002
- Equity / pot odds: 0.0 / 0.2767857142857143
- Action: **fold**; confidence: 0.85
- Situation: Heads-up preflop, out of position (index 0), hole 6s Jd offsuit, supplied equity 0.0 vs pot odds 0.277, pot 3.24, to_call 1.24.
- Rationale: Supplied equity (0.0) is below required pot odds (27.7%), so calling is -EV. Weak dominated offsuit hand out of position; fold preserves the 99.0 stack for better spots.
- Self model: Tight-aggressive; folds weak offsuit hands preflop when equity is below pot odds.
- Opponent model: Aggressive (aggression 1.0) and never folds preflop (fold rate 0.0); calling invites re-raise pressure, and a raise will not win the pot by folds (all-fold probability only 0.33).
- Risk flags: ['equity_estimate_uninitialized_at_0.0', 'weak_offsuit_hand', 'out_of_position']

## Hand 1 · preflop · vs tag

- Cards: `4s 9h`; board: `None`
- Pot / to call: 1.5 / 0.5
- Equity / pot odds: 0.0 / 0.25
- Action: **fold**; confidence: 0.9
- Situation: Preflop heads-up vs opponent's 1.74 open raise; holding 4s 9h, need 25% pot odds but equity_estimate is 0.0, far below breakeven.
- Rationale: Pot odds (0.25) vastly exceed supplied equity estimate (0.0); weak unsuited hole cards OOP to an aggressive opener make calling unprofitable and raising spew.
- Self model: Folding weak hands to large raises; defending only with speculative/strong holdings
- Opponent model: Aggressive opener who takes pots down uncontested; raising wide preflop and pressuring limpers
- Risk flags: ['equity_estimate_0.0', 'weak_hole_cards_unsuited', 'opponent_aggression_high']

## Hand 0 · preflop · vs tag

- Cards: `Qd Ah`; board: `None`
- Pot / to call: 1.5 / 0.5
- Equity / pot odds: 1.0 / 0.25
- Action: **raise**; confidence: 0.9
- Situation: Heads-up preflop, no community cards, hole Qd Ah (AQ offsuit), facing 0.5 to call into 1.5 pot (pot odds 0.25), equity estimate 1.0, stack 99.5.
- Rationale: AQ offsuit is a premium heads-up hand with high equity (1.0); pot odds 0.25 are easily covered, and raising builds the pot and pressures the opponent who shows moderate aggression/fold tendencies (0.5).
- Self model: Strong starting hand, in position index 1, healthy stack; aggressive preflop value strategy.
- Opponent model: Opponent aggression 0.5 and fold 0.5, roughly balanced; willing to call a moderate raise but may fold to pressure, so size around 3/4 pot to extract value.
- Risk flags: ['overcards-dominated-if-board-pairs-low', 'opponent-may-re-raise', 'equity-1.0-may-be-simulator-normalized-not-exact']

# Post-hand reflection examples

## Hand 0 · vs tag

- Outcome: None
- Review: None
- Worked: None
- Failed: None
- Belief updates: None
- Adjustment: None
- Calibration: None

## Hand 1 · vs tag

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

## Hand 1 · vs tag

- Outcome: None
- Review: None
- Worked: None
- Failed: None
- Belief updates: None
- Adjustment: None
- Calibration: None

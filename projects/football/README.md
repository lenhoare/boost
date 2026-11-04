# football

English Premier League 1X2 results, 2010-11 to the first 40 games of 2026-27.

Three questions (from the original brief):

1. Can we forecast results well?
2. Are our probabilities better than the bookmaker's?
3. How consistently, and is it bettable?

## Files

- `data/combined.csv` source; `data/data.md` the brief plus every cleaning decision and the leak audit.
- `data/prepare.py` cleaning, standard names, de-vigged book probabilities `book_h/d/a`.
- `features.py` pre-match features only (see the docstring for the rule). Prefixes `h_` home side, `a_` away side, `d_` home minus away.
- `evaluate.py` betting analysis on test rows: flat-stake ledger by EV threshold with bootstrap CI, breakdowns, edge reliability, fractional Kelly.
- `models/` one yaml per experiment. Copy one to start another.

## Models

| model | question |
|---|---|
| `result_1x2` | histories only, holdout 75/25. The pure "can we forecast" model. |
| `result_1x2_wf` | same, walk-forward by season. Out-of-sample every season from 2014-15. |
| `result_1x2_book` | book prices as ordinary features alongside histories. |
| `result_1x2_offset` | start from the book's log-odds, learn a correction. The direct test of question 2. |
| `result_1x2_offset_wf` | the offset test season by season. |
| `result_1x2_offset_wf_strict` | same, with two seasons for early stopping and heavier regularisation. |
| `result_1x2_offset_pre_wf` | strict offset from the Bet365 pre-match price in every season (a consistent, softer bar). |

| `result_1x2_dc_wf` | histories plus the Dixon-Coles goals model as features. |
| `result_1x2_dc_offset_wf` | start from Dixon-Coles, learn a correction from histories. |
| `result_1x2_offset_pre_dc_wf` | start from Bet365 pre-match, learn a correction from Dixon-Coles only. |
| `line_move_h`, `_d`, `_a` | line model: predict Pinnacle's move from the pre-match snapshot to the close. Evaluate with `analysis/line_eval.py`. |

`goals_model.py` is the Dixon-Coles model behind the `dc_*` features.
`analysis/blend.py` tests whether a forecast adds anything to a price;
`analysis/line_eval.py` turns the three line models into a predicted close
and tests it on outcomes and as a bet selector.

Every model has `seeds: 5` and a `compare:` block scoring it against Bet365,
Pinnacle and market-average prices, pre-match and closing, plus Betfair closing.
`measurement.md` has the results.

## Reading a run page

- Summary: `model_minus_baseline` negative means we beat the closing line on logloss.
- Calibration: probabilities should sit on the diagonal.
- By season: where any edge lives, and whether the Betfair era (2024-25 on, overround about 0.6%) differs from the Bet365 era (about 3 to 5%).
- Betting: `roi_ci_lo/hi` is the honest number. Odds are closing prices; getting the close on an exchange is optimistic.
- Edge reliability: if `observed` tracks `model_p` rather than `book_p` in the outer bins, the disagreement is information.

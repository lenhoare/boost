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

## Reading a run page

- Summary: `model_minus_baseline` negative means we beat the closing line on logloss.
- Calibration: probabilities should sit on the diagonal.
- By season: where any edge lives, and whether the Betfair era (2024-25 on, overround about 0.6%) differs from the Bet365 era (about 3 to 5%).
- Betting: `roi_ci_lo/hi` is the honest number. Odds are closing prices; getting the close on an exchange is optimistic.
- Edge reliability: if `observed` tracks `model_p` rather than `book_p` in the outer bins, the disagreement is information.

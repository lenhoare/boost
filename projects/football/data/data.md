# Data

## The brief (as given)

The combined.csv dataset should be fairly clean.

Training set: the top 75% of sensible data (earliest in time). Prematch
bookmakers odds will be evaluated against AND can be available to learn from in
this draft (we may disable if it dominates). Until 2024 Bet365 odds are
present which are harsh. Later Betfair odds are more realistic.

Evaluation set: the top 25%.

Most columns in this dataset should be included.

Moon phase: leave out for now.

## What prepare.py does

- Parses `date` (mixed dd/mm/yy and dd/mm/yyyy), sorts by date then teams, adds `match_id`.
- Renames `fthg/ftag` to `home_goals/away_goals`. Keeps `ftr` (H/D/A) as the 1X2 target. Adds `goal_diff`, `total_goals`.
- Turns the source's `-1` "not collected" into null for corners, cards, shots, fouls, possession. Turns `0.0` xG into null (early seasons have no xG; a true 0.00 does not occur).
- `attendance` "N/A" to null, cast to integer.
- De-vigs the closing 1X2 prices into `book_h`, `book_d`, `book_a` (1/odds normalised to sum to one) and keeps `book_overround`. Only rows with all three prices above 1.01 get book probabilities.
- Parses goal timings into `home_goals_ht`, `away_goals_ht` (minute 45 and stoppage count as first half), `first_goal_minute`, `first_goal_side`. Not used by the current models; there for later timing targets.
- Renames `home_prematch_xg/ppg` to `home_xg_pre`, `home_ppg_pre` (same for away).

## Sources

`combined.csv` is FootyStats (xG, pre-match xG/PPG, goal timings, venue,
attendance, referee) plus one football-data.co.uk price set per match. The full
football-data files are in `data/football-data/E0_<yy><yy>.csv` (downloaded
2026-11-04 from `https://www.football-data.co.uk/mmz4281/<yy><yy>/E0.csv`).
`prepare.py` joins them on (date, home, away). All 6120 rows match, and the
single `odds_home` price equals the football-data column its `odds_source`
names on 100% of rows: `B365H` to 2018-19, `B365CH` 2019-20 to 2023-24,
`BFECH` from 2024-25. That price is not a closing price before 2019-20:
football-data collects pre-match prices on Friday afternoon for weekend
games and Tuesday for midweek ones.

Extra columns from football-data (sets: `b365`, `b365c` Bet365 pre-match /
closing; `ps`, `psc` Pinnacle; `avg`, `avgc` market average; `max`, `maxc`
best available; `bfe`, `bfec` Betfair exchange):

- `odds_<set>_{h,d,a}` raw 1X2 prices; `book_<set>_{h,d,a}` de-vigged
  (proportional), `book_<set>_overround`. No `book_max*`: best price across
  books is not one market.
- `ou_<set>_{over,under}` over/under 2.5 goals prices; `p_<set>_over25` de-vigged.
- `ah_line`, `ah_line_c` Asian handicap line for the home side (pre-match,
  closing); `ah_<set>_{h,a}` prices.
- `kickoff` time, 2019-20 on.
- `bk_<code>_{h,d,a}` every individual bookmaker's 1X2 prices as football-data
  codes them (`b365`, `bw`, `wh`, `vc`, `lb`, `iw`, `ps`, `bmgm`, ...; a
  trailing `c` is the closing price, e.g. `bk_b365c_h`). Columns differ by season.
- `odds_bestuk_{h,d,a}` / `odds_bestukc_*` best pre-match / closing price over
  mainstream UK books (`UK_BOOKS` in `prepare.py`: no Pinnacle, no exchange, no
  offshore books), `n_bestuk` how many fed it. Use this rather than `max`: the
  football-data maximum often sums to under 100% (a guaranteed-profit
  arbitrage), a sign of stale or erroneous quotes. Best UK keeps a 1 to 3.7% margin.

Before 2019-20 `avg`/`max` are Betbrain's (`BbAvH`, `BbMxH`) and there are no
closing prices except Pinnacle. Prices at or below 1.01 are nulled.

| set | seasons present |
|---|---|
| `b365`, `avg`, `max`, O/U, AH | all |
| `ps`, `psc` | 2012-13 to mid 2025-26 (55% of 2025-26; Pinnacle feed stopped) |
| `b365c`, `avgc`, `maxc`, closing O/U and AH | 2019-20 on |
| `bfe`, `bfec` | 2024-25 on |

Bookmaker logloss over identical matches (lower is sharper):

| seasons | b365 | b365c | ps | psc | avg | avgc |
|---|---|---|---|---|---|---|
| 2012-13 to 2018-19 (2660) | 0.9494 | | 0.9488 | **0.9466** | 0.9493 | |
| 2019-20 to 2023-24 (1900) | 0.9586 | **0.9535** | 0.9578 | **0.9535** | 0.9576 | 0.9537 |
| 2024-25 to 2025-26 (760) | 0.9947 | 0.9904 | | | 0.9929 | **0.9893** |

Pre-match to close is worth 0.003 to 0.005 of logloss within every book. That
is the size of edge an early-price model is trying to find.

None of these columns are model features: `features.py` lists features
explicitly, and the de-vigged sets are named `book_*` so the default
`exclude: ["book_*"]` would catch them anyway.

## Leak audit

- `home_ppg` / `away_ppg`: constant within every team-season (340 of 340). That is the season-final points per game and would leak the season's results into every match. Dropped.
- `home_ppg_pre` / `away_ppg_pre`: not our season-to-date PPG (median gap 0.25) and not the last-5-at-venue figure either. Definition unknown. It correlates less with the result than our own season-to-date PPG (0.18 vs 0.24), so it is not a leak. Kept as a feature; low importance in practice.
- `home_xg`, shots, possession, corners, cards: post-match. Never used directly; features use them only through lagged team histories.
- `attendance`, `venue`: not used as features. `referee` is in the feature list but excluded by the default models.
- 2026-27 rows (40): results and odds present, all statistics null. They still get history features from prior seasons.

## Coverage by season

| seasons | odds | overround | xG | shots/possession/corners |
|---|---|---|---|---|
| 2010-11, 2011-12 | Bet365 | ~5.5% | none | none |
| 2012-13 to 2015-16 | Bet365 | ~2.6 to 4% | none | full |
| 2016-17 to 2018-19 | Bet365 | ~3% | full (pre-match xG 95%) | full |
| 2019-20 to 2023-24 | Bet365 closing | ~5.4% | full | full |
| 2024-25, 2025-26 | Betfair exchange closing | ~0.6 to 0.8% | full | full |
| 2026-27 (40 games) | Betfair exchange closing | ~0.6% | none yet | none yet |

The 25% holdout test starts in 2022-23 and so includes the whole Betfair era, where the book is a much sharper bar.

## Split

Holdout models: earliest 75% train (last 15% of that is the early-stopping slice), latest 25% test. Walk-forward models: retrain before each season from 2014-15 on, predict that season.

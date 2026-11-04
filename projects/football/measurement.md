# Measurement: are differences real, and against which price?

Step 1 of the improvement plan (2026-11-04). Core now reports, for every
comparison, a paired standard error and z from the per-match loss
differences, and with `seeds: N` the spread between single-seed runs.
Every model is scored against seven bookmaker bars (see `data/data.md`).

## Seed noise

Single-seed test logloss moves by about **0.0004 to 0.0005** between seeds
(0.0002 for the heavily regularised offset models). The last feature round
gained 0.0007 to 0.0009 from one seed each, so that gain was roughly 1.5 to 2
seed standard deviations: suggestive, not established. Every model now
averages 5 seeds.

## Results (walk-forward 2014-15 on unless noted; diff = model minus forecast logloss, ± one standard error)

| model | vs b365_pre | vs pinnacle_pre | vs pinnacle_close | vs b365_close | vs betfair_close |
|---|---|---|---|---|---|
| `result_1x2_wf` (histories only) | +0.0240 ± 0.0029 | +0.0245 ± 0.0030 | +0.0274 ± 0.0031 | +0.0315 ± 0.0039 | +0.0303 ± 0.0064 |
| `result_1x2_offset_wf` | −0.0003 ± 0.0013 | +0.0005 ± 0.0013 | +0.0034 ± 0.0013 | +0.0036 ± 0.0017 | +0.0003 ± 0.0007 |
| `result_1x2_offset_wf_strict` | −0.0010 ± 0.0011 | −0.0003 ± 0.0011 | +0.0026 ± 0.0010 | +0.0030 ± 0.0012 | +0.0012 ± 0.0014 |
| `result_1x2_offset_pre_wf` | +0.0018 ± 0.0007 | +0.0026 ± 0.0008 | +0.0054 ± 0.0013 | +0.0077 ± 0.0018 | +0.0057 ± 0.0029 |

Rows: b365_pre 4600, pinnacle 4390 (to mid 2025-26), b365_close 2700 (2019-20 on), betfair_close 778.

## What this says

1. **Histories alone are about 0.025 behind even the early prices** (z around +8).
   The gap is not noise and not a matter of which bookmaker is used.
2. **A learned correction on top of a price makes it worse.** Against its own
   starting price the correction costs 0.0018 to 0.0025 (z +2.5), whether the
   start is the mixed `book_*` column or a consistent Bet365 pre-match price,
   and with or without heavy regularisation. The team-history features carry
   no information the price lacks; the booster fits noise.
3. **The damage is concentrated in unusual seasons.** In thousandths of
   logloss, strict offset vs its start: 2015-16 +3.4 (Leicester), 2019-20
   +4.5 and 2020-21 +14.3 (Covid, empty stadiums), 2018-19 −2.2, every other
   season within ±2. A correction learned from normal seasons misfires when
   the game itself changes.
4. **The offset in `result_1x2_offset_wf` switches price type in 2019-20**
   (Bet365 pre-match to closing, then Betfair closing). That is a flaw in the
   design of that test, but fixing it (`result_1x2_offset_pre_wf`) did not
   change the answer.

## Implication for the next steps

More Elo and form features of the same kind will not close the gap: the
market already has them. The candidates are information of a different kind
(a goals model with its own structure, line movement, team news) and softer
targets (the pre-match price, over/under, specific segments).

# Step 2: Dixon-Coles goals model (2026-11-04)

`goals_model.py`, refitted before every match date on earlier matches only.
Half-life 500 days and ridge 5 were chosen on 2011-12 to 2013-14 (logloss was
flat across half-lives of 1 to 2 years and ridge 3 to 10), so no test season
influenced them. `dc_*` is fitted on goals, `dcx_*` on a 50/50 goals/xG blend.

## On its own (walk-forward test seasons, 4600 matches)

| forecast | 1X2 logloss | vs b365_pre | vs pinnacle_close |
|---|---|---|---|
| Dixon-Coles, goals (`dc`) | **0.9758** | +0.0140 ± 0.0024 | +0.0176 ± 0.0027 |
| Dixon-Coles, 50% xG (`dcx`) | 0.9773 | +0.0155 ± 0.0024 | +0.0196 ± 0.0027 |
| Dixon-Coles, 80% xG | 0.9819 | +0.0201 ± 0.0027 | +0.0246 ± 0.0029 |
| `result_1x2_wf` (boosted histories) | 0.9858 | +0.0240 ± 0.0029 | +0.0274 ± 0.0031 |

The goals model is the best standalone forecaster here, 0.010 better than
the boosted model and 40% of the way from it to the bookmakers. Over/under 2.5
is also behind the market average (+0.006 to +0.009, z 3 to 5). xG helps a
little on totals and hurts on 1X2.

## Boosting with it

| model | test logloss | vs its own start / vs DC |
|---|---|---|
| `result_1x2_dc_wf` (histories + DC features) | 0.9849 | +0.0091 ± 0.0025 vs DC alone (z +3.6) |
| `result_1x2_dc_offset_wf` (DC + learned correction) | 0.9771 | +0.0013 ± 0.0011 vs DC (z +1.1) |
| `result_1x2_offset_pre_dc_wf` (Bet365 pre + correction from DC) | 0.9624 | +0.0006 ± 0.0003 vs Bet365 pre (z +1.7) |

Adding DC to 160 other features makes the booster *worse* than DC alone: a
good signal gets diluted. Correcting DC with team histories does not help.

## Does it add anything to a price? (`analysis/blend.py`)

Walk-forward two-weight blend, p ∝ price^a × DC^b, refitted each season:

| price | n | blend − price | z | mean weight on DC |
|---|---|---|---|---|
| b365_pre | 4600 | +0.0005 ± 0.0003 | +1.6 | −0.03 |
| pinnacle_pre | 4390 | +0.0006 ± 0.0004 | +1.5 | −0.00 |
| avg_pre | 4600 | +0.0004 ± 0.0005 | +0.8 | −0.06 |
| pinnacle_close | 4390 | +0.0006 ± 0.0004 | +1.3 | −0.02 |
| betfair_close | 398 | −0.0012 ± 0.0013 | −0.9 | −0.09 |

The weight on DC is about zero and usually slightly negative: where DC
disagrees with the market, the market is right. DC also does not predict
which way prices move before kick-off (correlation with the pre-to-close
move 0.00 to 0.06). **A good public-data goals model adds nothing to even
the early price.**

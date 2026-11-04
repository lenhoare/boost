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

# Step 3: the line model (2026-11-04)

football-data takes every bookmaker's pre-match price in one snapshot (Friday
afternoon for weekend games, Tuesday for midweek) and the closing prices at
kick-off. Pinnacle has both from 2012-13 to mid 2025-26 (5150 matches). How
far does the price move, can we predict it, and is that worth anything?

## What moves prices (correlation of each gap with the later move, H/D/A pooled)

| move of | predictor (pre-match gap) | corr | share of gap closed |
|---|---|---|---|
| Bet365 | market average − Bet365 | +0.25 | 64% |
| Bet365 | Pinnacle − Bet365 | +0.22 | 47% |
| Pinnacle | market average − Pinnacle | +0.11 | 32% |
| Pinnacle | Dixon-Coles − Pinnacle | +0.02 | 1% |
| market average | Pinnacle − average | +0.01 | 3% |

Soft books move towards the consensus; the consensus does not follow
Pinnacle; our goals model predicts nothing. Typical move in the home-win
probability: sd 2.6 points, more than 2 points in 40% of matches.

## Line model (`line_move_{h,d,a}`, walk-forward from 2014-15, 4390 matches)

Inputs (`ln_*` in `features.py`): every price in the snapshot relative to
Pinnacle's, the spread across books, margins, kick-off hour, Dixon-Coles.
Target: log(Pinnacle close) − log(Pinnacle pre) per outcome.

- Out-of-sample R² of the move: home 0.1%, draw 1.0%, away 3.5%. A simple
  linear rule on the market-average gap does about as well. Pinnacle's move is
  almost entirely news that arrives after the snapshot (team news, lineups).
- Predicted close on outcomes: 0.9567 against Pinnacle pre 0.9568 and
  Pinnacle close 0.9539. It recovers 5% of the pre-to-close improvement (z −1.0).

## As a bet selector (`analysis/line_eval.py`)

Back the best-EV side at a pre-match price when EV > threshold. CLV is the
expected profit if Pinnacle's close is the truth; it is far less noisy than
results. The proportional de-vig overstates longshots, which inflates both EV
and CLV on long odds, so every result was re-checked with the power de-vig.
Only results that survive both are listed as real.

| bet at | fair price | thr | bets | CLV (prop) | CLV (power) | ROI (power) |
|---|---|---|---|---|---|---|
| best UK pre | Pinnacle pre | 0.02 | 654–1081 | +4.8% [4.0, 5.5] | +2.8% [2.0, 3.5] | +0.9% [−14, +18] |
| best UK pre | **predicted close** | 0.02 | 571–978 | +5.2% [4.4, 6.0] | **+3.9% [3.1, 4.7]** | +7.9% [−8, +25] |
| best UK pre | Pinnacle pre | 0.05 | 182–473 | +7.0% [5.7, 8.4] | +4.6% [2.9, 6.3] | +9.5% [−27, +52] |
| best UK pre | **predicted close** | 0.05 | 155–377 | +8.4% [6.9, 10.0] | **+5.8% [4.2, 7.5]** | +14% [−27, +64] |
| Bet365 pre | predicted close | 0.00 | 550–813 | +2.1% [1.4, 2.8] | +0.4% [−0.2, +1.0] | +6.3% [−9, +22] |
| Pinnacle pre | predicted close | 0.02 | 3–98 | +4.9% [2.2, 7.7] | −3.3% (3 bets) | — |

(Bet counts differ between de-vig methods because the de-vig changes which
sides show value.)

## What this says

1. **Real but modest: shopping the best UK price against Pinnacle's snapshot.**
   Positive CLV in every era and under both de-vig methods; the line model
   adds about 1 point of CLV on top of the plain rule. Returns are positive
   but not yet distinguishable from zero: a few hundred bets a season at odds
   of 5 to 8 need several seasons to confirm a 4% edge.
2. **Not real: betting Bet365 alone, or betting Pinnacle itself.** Both looked
   good under the proportional de-vig and vanished under the power de-vig.
   The Pinnacle "edge" was longshot mispricing in the measurement, not in
   the market.
3. **The edge is shrinking.** Plain-rule CLV at best UK price (power de-vig)
   fell from +1.6% in 2012-16 to +0.6% in 2022-26.
4. **Practical caveats.** UK books restrict accounts that consistently beat
   the close, and the snapshot time is football-data's, not one we can choose.
   Collecting our own timestamped prices (and team news) is the way to make
   this sharper and to test it live.

# Step 3b: the Betfair exchange (2026-11-04)

Exchanges do not restrict winning accounts, so an edge there would be one
that lasts. `analysis/exchange_eval.py`: back the best net-EV side at the
exchange's pre-match back price, commission 2% or 5% on winnings, 778
matches from 2024-25 (Pinnacle on 590 of them). football-data gives back
prices only, so laying is not tested.

- **The exchange snapshot is already near-fair**: back prices sum to 100.65%.
- **No edge.** CLV against the exchange's own close is negative in every
  combination (fair = Pinnacle pre, market average or the line model's
  predicted close; 2% or 5% commission; either de-vig), typically −1.5% to
  −4%. CLV against Pinnacle's close is positive only under the proportional
  de-vig (the longshot artefact) and negative under the power de-vig.
  Realised ROI is negative throughout (e.g. −21% [−36, −6] for Pinnacle-fair
  bets at 2% commission, power de-vig).
- **The exchange is a good replacement reference** now Pinnacle's feed has
  stopped: on the shared 590 matches its pre-match price scores 0.9757
  against Pinnacle's 0.9769 (−0.0012 ± 0.0009).
- **But with the exchange as the fair price, best UK prices rarely show
  value**: 103 bets in 778 matches at EV > 0, CLV +0.5% [−1.1, +2.0]. The
  soft-book edge of earlier seasons has largely closed in the exchange era.

Taken together with step 3: the market information in a once-a-day snapshot
is fully priced by the sharp end of the market (Pinnacle, the exchange).
The only measurable value was soft UK books lagging that sharp price, and it
has been fading.

# Step 4: tuning the goals model (2026-11-04)

Is the Dixon-Coles model as good as it can be with this data?
`analysis/goals_tune.py` fits a grid fixed in advance:

- half-life 250 / 500 / 1000 days
- ridge 2 / 5 / 10
- home advantage: global, or dynamic (its own 120-day decay), or dynamic plus per team
- xG weight in the target 0 / 0.25 / 0.5
- low scores: rho only, or rho plus draw inflation

That is 162 variants, all walk-forward. Then **nested selection**: for each
test season, use whichever variant scored best on the seasons before it. The
score therefore pays for the choosing.

For speed the grid refits weekly instead of before every match date, and a
new season always triggers a refit. Checked on the current model: daily
0.97582, weekly 0.97597 (+0.00015 ± 0.00014). Draw-inflated: daily 0.97570,
weekly 0.97586. Weekly is as good, and about 3 times faster.

Test seasons 2014-15 on, 4600 matches; current model (hl500, r5, global, xg0,
rho) 0.9760 weekly:

| | logloss | vs current |
|---|---|---|
| nested selection (honest) | 0.9756 | −0.0003 ± 0.0007 (z −0.5) |
| best in hindsight (hl250 r2 global xg0.25 draw-infl.) | 0.9737 | −0.0023 ± 0.0010 (optimistic) |

One ingredient changed at a time:

| change | diff ± se |
|---|---|
| half-life 250 | −0.0010 ± 0.0009 |
| half-life 1000 | +0.0036 ± 0.0011 |
| ridge 2 / 10 | +0.0004 / +0.0003 ± 0.0004 |
| dynamic home | +0.0005 ± 0.0005 |
| dynamic + per-team home | +0.0001 ± 0.0006 |
| xG weight 0.25 | −0.0002 ± 0.0006 |
| xG weight 0.5 | +0.0015 ± 0.0013 |
| draw inflation | −0.0001 ± 0.0004 |

Dispersion: Pearson variance/mean given the model's means is 0.991 for home
goals and 1.042 for away goals. Poisson fits, so a negative-binomial model
is not worth building.

What this says:

- **The current model is already at the top of this family.** Nothing beats
  it beyond noise. Nested selection gains 0.0003. Even the best variant in
  hindsight gains only 0.002, against a gap to the pre-match prices of about
  0.02.
- **Per-team and time-varying home advantage add nothing.** Home advantage did
  drop in the empty-stadium season, 2020-21, but a single global term refitted
  weekly already follows it closely enough.
- **xG helps a little at most, and only blended lightly (0.25).** Half xG
  is worse. Nested selection settles on xg0.25 and a 250-day half-life from
  2022-23 onwards, and gains in 2024-25 (−0.0085) and 2025-26 (−0.0038). It
  may be that recent seasons reward faster updating. Two seasons are not
  enough to rely on, but this is the setting to watch.
- **Decision:** keep the current settings in `project.yaml`. The scale of
  these differences says a better goals model alone will not close the gap
  to the bookmakers. Any edge has to come from information the model lacks.

# Step 5: a few features on top of the goals model (2026-11-04)

`analysis/feature_blend.py` uses a multinomial logit that keeps the DC
log-probabilities as an offset with a weight of their own. On top it puts a
home and an away logit (draw is the reference), and four home-minus-away
features, standardised and lightly shrunk (L2):

- FootyStats pre-match xG
- Elo
- rest days, clipped to ±7
- promoted

Fitted before each season on all earlier ones; test 2014-15 on. Each feature
is compared against the same model with no features ("recalibrated DC"), so
any gain belongs to the feature.

A. Goals model plus features, no prices (4600 matches):

| | logloss | vs recalibrated DC |
|---|---|---|
| raw DC | 0.9758 | |
| recalibrated DC | 0.9767 | (+0.0009 ± 0.0004 vs raw) |
| + pre-match xG | 0.9757 | −0.0010 ± 0.0014 |
| + Elo | 0.9755 | −0.0012 ± 0.0011 |
| + rest days | 0.9769 | +0.0003 ± 0.0001 |
| + promoted | 0.9775 | +0.0008 ± 0.0005 |
| + all four | 0.9758 | −0.0009 ± 0.0018 |

Elo and pre-match xG perhaps add 0.001 to the goals model. That is within
noise, and no better than the raw DC forecast. Rest days and the promoted
flag only hurt, because DC already handles promoted clubs through its
newcomer prior.

B. Price + DC + features, against the price alone, recalibrated the same way:

| price | price | + DC | + DC + all four |
|---|---|---|---|
| Bet365 pre (4600) | 0.9624 | +0.0003 ± 0.0002 | +0.0023 ± 0.0010 |
| Pinnacle pre (4390) | 0.9576 | +0.0003 ± 0.0002 | +0.0026 ± 0.0011 |
| market average pre (4600) | 0.9615 | +0.0003 ± 0.0002 | +0.0022 ± 0.0010 |

Every feature makes the price worse, at every shrinkage level. Results with
Pinnacle, all four features:

| L2 | diff ± se |
|---|---|
| 1 | +0.0026 ± 0.0011 |
| 10 | +0.0018 ± 0.0009 |
| 100 | +0.0006 ± 0.0005 |

Each feature alone is ≥ 0 too. The weights that are fitted point the "wrong"
way, for example a higher home Elo gap making an away win *more* likely than
the price says. They also wander from season to season. That is how noise
around a price looks when the price already contains the information.

What this says: the obvious public information is priced in, whether it
comes from goals, xG, ratings, rest or promotion. A history-only model,
however it is built, sits about 0.018 behind the Friday price and adds
nothing to it. Beating the bookmakers with this data would need something the
market does not have, or does not have yet, at the moment of betting. The
candidates are earlier timestamps than Friday (the Monday-to-Friday question
in step 3) and team news.

# Step 6: picking bets instead of pricing every match (2026-11-04)

Logloss asks "would I make a better bookmaker?". A punter only needs to be
right on the bets they choose. A model can be worse on average and still be
right where it disagrees strongly with the price. `analysis/bet_select.py`
tests that directly, on walk-forward forecasts from 2014-15 (4600 matches):

- dc: Dixon-Coles
- dcx: DC with half xG
- dcf: DC + Elo + pre-match xG, from step 5

**Who is right when they disagree?** Here q is the model probability and m
the market's (Pinnacle pre-match, power de-vig). Regress (result − m) on
(q − m) through the origin, over every match and outcome. A slope of 0 means
the market is right; 1 means the model is.

| model | slope ± se | |q − m| ≥ 0.10 only |
|---|---|---|
| dc | +0.01 ± 0.06 | +0.01 ± 0.09 |
| dcx | +0.00 ± 0.06 | −0.09 ± 0.09 |
| dcf | −0.02 ± 0.07 | −0.03 ± 0.11 |

At most about 14% of the model's disagreement is real (upper 95% bound), and
the best estimate is none. The biggest disagreements are no better. No
pre-chosen segment stands out:

- home / draw / away
- favourites / middle / outsiders
- early season / rest of season
- promoted club involved / not

The highest was "promoted club involved" at +0.20 ± 0.12 for dc and dcf.
That is one of 14 segments × 3 models, so it is in line with chance. As bets
(below) it has CLV −3.3% [−3.9, −2.8].

**As bets.** Back every outcome with model EV > threshold:

| model | bet at | EV > 0.05 bets | CLV vs Pinnacle close | ROI |
|---|---|---|---|---|
| dc | Bet365 pre | 3849 | −6.3% [−6.6, −6.0] | −2.0% [−8.8, +4.7] |
| dc | best UK pre | 4349 | −3.7% [−4.0, −3.3] | +0.6% [−5.7, +7.1] |
| dc | Pinnacle pre | 4284 | −4.6% [−4.9, −4.3] | −1.7% [−8.1, +4.7] |
| dcf | best UK pre | 3845 | −3.4% [−3.7, −3.1] | −1.3% [−6.7, +4.9] |
| reference: Pinnacle pre as fair, best UK, EV > 0.02 | | 671 | +2.9% [+2.1, +3.6] | +1.8% [−13, +18] |

CLV is clearly negative for every model, every price and every threshold
(0, 0.05, 0.10, 0.20). Raising the threshold makes it slightly *worse*: the
more confident the model is against the price, the more the price moves
against the bet. ROIs straddle zero only because results are noisy. Thousands
of bets with CLV at −3% to −6% is a reliable loser over time.

What this says: being selective does not rescue the history models. The
price has already moved away from the matches the model likes, and the
closing price confirms the market. Information in the football data alone
gives no edge to pick bets with. The only positive line is still the step 3
rule: shopping soft UK prices against the sharp price, with no model needed.

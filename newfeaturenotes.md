# Adding features: the process, and a log of every round

The recipe is the same every time. A feature round touches one project file,
one optional config line, and produces new runs. Core is never edited.

## The recipe

1. **Decide the name.** Follow the prefix scheme so tooltips and diffs come for
   free: `h_` home side, `a_` away side, `d_` home minus away, anything else
   is match-level. A statistic that already has a home and away source column
   is one line in `STATS` at the top of `projects/football/features.py` and
   gets every window automatically.
2. **Compute it in `projects/football/features.py`.**
   - Lagged per-team aggregates go in `_team_history` (they are shifted by one
     row per team, so nothing from the current match leaks in).
   - Anything with running state (ratings, league table, head-to-head) goes in
     the `_sequential` loop: read the state for the row *first*, update it
     *after*.
   - Differences and match-level columns go in `build`.
3. **Add it to the feature list** at the bottom of `build`. Columns starting
   `h_`, `a_`, `d_` are picked up by pattern; anything else is listed by name.
4. **Describe it.** Derived names get their text from `describe`; one-off names
   go in the `_WHOLE` dictionary. `boost report` warns on the run page if a
   feature has no description.
5. **Any knob goes in `projects/football/project.yaml`** under `features:` and
   is read in `build` with a default.
6. **Rebuild:** `.venv/bin/boost -p football features --force`. Check the
   printed null rates and eyeball a few rows for one team.
7. **Retrain the models you want to compare:**
   `.venv/bin/boost -p football train -m result_1x2_wf --tag "what changed"`.
   Earlier runs of the same yaml are the before; the new run is the after. Both
   stay on the runs index. If you want "without" and "with" from the same
   moment, make a second yaml that excludes the new columns by pattern.
8. **Commit** with a message saying what was added and why.

## Log

### 2026-09-18 — venue-specific result and goals ratings

**Why.** One Elo per team blends home and away strength. Six ratings per team
separate them: results at home, results away, goals scored at home, goals
scored away, goals conceded at home, goals conceded away.

**What was added** (12 features, all in `_sequential`):

| feature | meaning |
|---|---|
| `h_elo_res_home` | home side's Elo from home results only (start 1500, step `elo_k`) |
| `a_elo_res_away` | away side's Elo from away results only |
| `elo_venue_exp_home` | home-win probability implied by those two |
| `h_elo_att_home`, `h_elo_def_home` | home side's goals scored / conceded ratings at home |
| `a_elo_att_away`, `a_elo_def_away` | away side's goals scored / conceded ratings away |
| `elo_xg_home`, `elo_xg_away` | expected goals implied by the goals ratings |
| `d_elo_res_venue` | home-results Elo minus away-results Elo |
| `d_elo_attack_h`, `d_elo_attack_a` | attack rating minus the opposing defence rating, each direction |
| `d_elo_xg` | expected goals, home minus away |

**Mechanism.** Result ratings are ordinary Elo with no home-advantage term (the
venue split carries it). Goals ratings are in log units, starting at 0:
expected home goals = running league-average home goals × exp(home attack −
away defence). After the match the attack rating rises and the defence rating
falls by `elo_goals_k` × (actual − expected). Ratings carry across seasons
unchanged; promoted clubs start at the mean. `elo_goals_k: 0.05` was added to
`project.yaml`.

**Runs to compare.** `result_1x2` and `result_1x2_wf` before (runs from 12:47)
and after (tagged "plus venue and goals Elos").

**Result.** Test logloss, book at 0.9605 (holdout) / 0.9590 (walk-forward):

| model | before | after | change |
|---|---|---|---|
| `result_1x2` | 0.9916 | 0.9907 | −0.0009 |
| `result_1x2_wf` | 0.9867 | 0.9860 | −0.0007 |

Small but in the right direction on both splits. In the walk-forward run the top features by gain are now:

- 1. `d_elo` (20.8%)
- 2. `elo_exp_home` (14.2%)
- 3. `d_elo_xg` (7.0%)
- 4. `d_std_gd` (2.4%)
- 5. `d_elo_res_venue` (2.2%)
- 6. `h_prev_gd` (1.7%)
- 7. `d_ewm_gf` (1.6%)
- 8. `a_prev_gd` (1.6%)

New features by rank:

- 3. `d_elo_xg` (7.0%)
- 5. `d_elo_res_venue` (2.2%)
- 9. `elo_venue_exp_home` (1.5%)
- 21. `elo_xg_away` (0.7%)
- 26. `d_elo_attack_a` (0.6%)
- 28. `h_elo_att_home` (0.6%)
- 37. `a_elo_def_away` (0.5%)
- 41. `a_elo_res_away` (0.5%)
- 43. `elo_xg_home` (0.5%)
- 47. `d_elo_attack_h` (0.4%)
- 56. `h_elo_res_home` (0.4%)
- 59. `a_elo_att_away` (0.4%)
- 74. `h_elo_def_home` (0.3%)

The venue-split result Elo and the goals-based expected goals sit just behind the overall Elo. They carry much the same signal, which is why the gain is modest; the gap to the bookmaker is still about 0.03.

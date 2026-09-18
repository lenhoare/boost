# boost

A reusable gradient-boosting framework. Core knows tabular data, an ordering, a
target, a loss and a boosting library. It never knows what a row means.
Projects supply the data recipe, the feature recipe, model configs and optional
evaluation hooks.

```
boost/                 core package (no project knowledge)
projects/<name>/
  project.yaml         data/features/evaluation modules, column roles
  data/                source files, prepare.py, cached prepared.parquet + features.parquet
  features.py          build(df, cfg) -> (frame, feature_names)
  evaluate.py          optional: evaluate(ctx) -> report sections
  models/<model>.yaml  one training recipe: task, features, split, booster params
  runs/<run_id>/       predictions, curve, importance, model, report page
```

## Commands

```bash
.venv/bin/boost -p football prepare                 # data recipe -> prepared.parquet, then features
.venv/bin/boost -p football features [--force]      # rebuild / summarise the feature frame
.venv/bin/boost -p football models                  # list model configs
.venv/bin/boost -p football train -m result_1x2 [--tag "note"]
.venv/bin/boost -p football report [--run latest]   # re-render a run page (re-runs evaluation hooks)
.venv/bin/boost -p football runs                    # list runs, refresh runs/index.html
```

Open `projects/<name>/runs/index.html` for the run table; each run has its own page.

## The three layers

**Core** (`boost/`): splits (holdout, walk-forward by period, random), LightGBM
training with early stopping and optional refit, proper scores (logloss, Brier,
calibration, RMSE, Poisson deviance), baseline comparison, feature importance,
run store, HTML report. Task types: `binary`, `multiclass`, `regression`, `poisson`.

**Project** (`projects/<name>/`): `data/prepare.py` turns raw files into one
clean frame. `features.py` adds predictor columns and names them. Nothing in
core reads a project column by name; `project.yaml` `roles:` says which column
is the time order, which is the period, which identify a row.

**Evaluation hooks** (`evaluate.py`): project-specific analysis that core has
no business knowing. The football project's hook is the betting analysis.
A project without one still gets the core sections.

## Model config

```yaml
task: {type: multiclass, target: ftr, classes: [H, D, A]}
baseline: [book_h, book_d, book_a]     # probability columns to compare against (optional)
offset: true                           # start the booster from the baseline, learn a correction (optional)
passthrough: [odds_home, ...]          # columns copied into predictions.parquet for hooks
rows: "season_start >= 2012"           # optional SQL filter on the feature frame
features: {include: ["*"], exclude: ["book_*"]}   # globs over the feature names
split: {strategy: holdout | walk_forward | random, ...}
model: {rounds: 3000, early_stopping: 150, refit: true, params: {...lightgbm params...}}
```

`offset: true` is the sharpest way to ask "does my information add anything to
an existing forecast?": the booster starts from the baseline's log-odds and can
only improve on it. If early stopping fires at round one, the answer is no.

Walk-forward retrains before each period on everything before it and predicts
that period. Predictions across folds form one out-of-sample series over most
of the data, which is what a betting ledger needs.

## Adding a project

Copy the folder layout, write `prepare.py` and `features.py`, set `roles:`,
add one model yaml. No core edits should be needed. If one is, it belongs in
core only if a second project would need it too.

## Environment

Python 3.12, `uv venv .venv && uv pip install -e .`. LightGBM on CPU; these
datasets train in seconds. GPU builds of LightGBM/XGBoost are a swap inside
`boost/train.py`, the only file that imports the boosting library.

"""Plain-English definitions for terms that appear in run pages.

Keys are matched (case-insensitively) against table headings and against the
label in the first column of each row. Projects add their own terms by
defining GLOSSARY in their evaluation module.
"""

GLOSSARY: dict[str, str] = {
    # splits
    "train": "Rows the model learned from. Scores here are flattering because the model has seen the answers.",
    "valid": "A slice held back from the training rows and used only to decide when to stop adding trees (early stopping). The model never fits to it, but it did influence one choice, so it is slightly less honest than test.",
    "test": "Rows never used in training in any way. This is the honest score.",
    "fold": "One train / valid / test division. A holdout run has one fold. A walk-forward run has one fold per period, each retraining on everything before it.",
    "holdout": "One split: earliest rows train, latest rows test.",
    "walk_forward": "Retrain before each period on everything earlier, predict that period, move on. The test predictions from all folds together form one out-of-sample series.",
    "n_train": "Number of rows the model learned from in this fold.",
    "n_valid": "Number of rows used for early stopping in this fold.",
    "n_test": "Number of rows scored out-of-sample in this fold.",
    "best_iteration": "How many trees were kept. Training stops when the valid loss has not improved for a while and rewinds to the best point. A value of 1 means the first tree did not help: the model found nothing to add.",
    "valid_loss": "The loss on the valid rows at the best iteration.",
    # scores
    "metric": "The name of the score. Each has a definition in the tooltip on its row.",
    "n": "Number of rows scored.",
    "logloss": "Average surprise at what actually happened: minus the log of the probability the model gave to the real outcome, averaged over rows. Lower is better. Guessing three outcomes equally scores 1.099; a perfect forecaster scores 0.",
    "brier": "Average squared distance between the forecast probabilities and the outcome (1 for what happened, 0 for what did not). Lower is better. Less sensitive than logloss to confident mistakes.",
    "accuracy": "How often the most likely outcome was the actual one. Ignores how confident the forecast was, so it is a weaker guide than logloss.",
    "baseline_n": "Rows where the comparison forecast was available.",
    "baseline_logloss": "The logloss of the comparison forecast on the same rows. Here that is the bookmaker's closing prices with the margin removed.",
    "model_logloss_on_baseline_rows": "The model's logloss restricted to rows where the comparison forecast exists, so the two are compared like for like.",
    "model_minus_baseline": "Model logloss minus baseline logloss. Negative means the model beat the comparison; positive means it lost. A few thousandths either way is noise.",
    "diff_se": "Standard error of the logloss difference, computed from the per-row differences between the two forecasts. The gap would typically move by about this much with a different sample of matches.",
    "diff_ci_lo": "Low end of a 95% interval for the logloss difference (mean minus 1.96 standard errors).",
    "diff_ci_hi": "High end of a 95% interval for the logloss difference. If the interval contains 0, the data cannot tell the two forecasts apart.",
    "diff_z": "The logloss difference divided by its standard error. Below about -2: the model is reliably better. Above about +2: reliably worse. In between: could be luck.",
    "diff": "Model logloss minus the comparison forecast's logloss on the rows they share. Negative means the model was better.",
    "forecast": "A comparison forecast named in the model config's compare block, such as a bookmaker's de-vigged prices.",
    "model_logloss": "The model's logloss on the rows where this comparison forecast exists.",
    "forecast_logloss": "The comparison forecast's logloss on the same rows.",
    "seed_logloss_mean": "With several seeds, each seed's model is also scored alone. This is their average test logloss; the headline logloss scores the averaged predictions, which is usually a little better.",
    "seed_sd": "Standard deviation of test logloss across single-seed models. An improvement smaller than this from one run to the next is indistinguishable from changing the random seed.",
    "seed_logloss_min": "Best test logloss among the single-seed models.",
    "seed_logloss_max": "Worst test logloss among the single-seed models.",
    "seed_minus_baseline_mean": "Average of model_minus_baseline over the single-seed models.",
    "best_iteration_min": "Fewest trees kept by any seed in this fold.",
    "best_iteration_max": "Most trees kept by any seed in this fold.",
    "rmse": "Root mean squared error, in the target's units. Lower is better.",
    "mae": "Mean absolute error, in the target's units. Lower is better.",
    "poisson_deviance": "The proper score for count targets; lower is better.",
    # calibration
    "class": "One possible outcome.",
    "bin": "Rows grouped by the probability the model gave this class, in steps of 0.1.",
    "mean_pred": "Average probability the model gave this class within the bin.",
    "observed": "How often this class actually happened within the bin. If the model is honest this matches mean_pred.",
    "gap": "observed minus mean_pred. Positive: the model was under-confident here; negative: over-confident.",
    "observed_rate": "How often this outcome happened across all test rows.",
    "picked_rate": "How often the model made this outcome its most likely one. Compare with observed_rate: a big difference means the model rarely picks that outcome even when it happens.",
    # periods
    "period": "The breakdown unit (here, a season).",
    # importance
    "feature": "One input column. See the project's features.py docstring for naming.",
    "gain": "Total reduction in training loss credited to splits on this feature. Bigger means the model leaned on it more.",
    "split": "Number of tree splits that used this feature (averaged over folds).",
    "folds": "How many folds this feature appeared in.",
    "gain_share": "This feature's gain as a fraction of all gain, so it sums to 1 down the table.",
}

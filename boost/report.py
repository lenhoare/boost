"""Run pages and the run index. Core sections plus whatever the project's hooks add."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from boost.glossary import GLOSSARY as CORE_GLOSSARY
from boost.metrics import calibration_table, class_balance, comparison_by_period, comparison_table, period_table
from boost.project import Project
from boost.store import list_runs, load_meta, load_predictions
from boost.types import RunContext, Section, Task

PALETTE = ["#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8"]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    })


def save_fig(fig: plt.Figure, run_dir: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(run_dir / name, bbox_inches="tight")
    plt.close(fig)
    return name


# --- core sections -----------------------------------------------------

def _summary_section(meta: dict[str, Any], curve: pl.DataFrame, task: Task) -> Section:
    metrics = meta.get("metrics") or {}
    keys = list(dict.fromkeys(k for split in metrics.values() for k in split))
    rows = [{"metric": k, **{split: metrics[split].get(k) for split in metrics}, "meaning": METRIC_MEANING.get(k, "")} for k in keys]
    folds = pl.DataFrame(meta.get("folds") or [])
    notes = []
    if meta.get("description"):
        notes.append(str(meta["description"]))
    if meta.get("tag"):
        notes.append(f"Tag: {meta['tag']}")
    notes += [
        f"Task: {task.type} on `{task.target}`" + (f", classes {list(task.class_names)}" if task.class_names else ""),
        f"Rows {meta.get('rows')}, features {meta.get('n_features')}, folds {len(meta.get('folds') or [])}, {meta.get('seconds')} s.",
    ]
    captions = {
        "Metrics": "One row per score, one column per split. `valid` rows were held back from training to decide when to stop adding trees; `test` rows were never seen at all and are the honest number. Hover any heading or row label for a definition.",
        "Folds": "How the rows were divided. A holdout run has one fold; a walk-forward run has one per period, each retrained on everything before it. `best_iteration` is how many trees were kept; 1 means the model found nothing worth adding.",
    }
    return Section("Summary", {"Metrics": pl.DataFrame(rows), "Folds": folds}, [], notes + _how_to_read(task, bool(meta.get("baseline"))), captions)


METRIC_MEANING = {
    "n": "rows scored",
    "logloss": "lower is better; 1.099 is guessing three ways equally",
    "brier": "lower is better; 0 is perfect",
    "accuracy": "higher is better, but ignores confidence",
    "baseline_n": "rows where the comparison forecast exists",
    "baseline_logloss": "the comparison forecast's logloss; lower is better",
    "model_logloss_on_baseline_rows": "our logloss on those same rows",
    "model_minus_baseline": "negative = we beat the comparison; positive = it beat us",
    "diff_se": "standard error of model_minus_baseline, from the per-row differences",
    "diff_ci_lo": "95% interval, low end, for model_minus_baseline",
    "diff_ci_hi": "95% interval, high end; if the interval spans 0 the gap could be luck",
    "diff_z": "model_minus_baseline / diff_se; below -2 we are reliably better, above +2 reliably worse",
    "seed_logloss_mean": "average test logloss of the single-seed models (the reported logloss averages their predictions)",
    "seed_sd": "spread of test logloss between single-seed runs: changes smaller than this are seed noise",
    "seed_logloss_min": "best single-seed test logloss",
    "seed_logloss_max": "worst single-seed test logloss",
    "seed_minus_baseline_mean": "average model_minus_baseline of the single-seed models",
    "rmse": "lower is better, target units",
    "mae": "lower is better, target units",
    "poisson_deviance": "lower is better",
}


def _how_to_read(task: Task, has_baseline: bool) -> list[str]:
    """Plain-English definitions of the numbers in the metrics table."""
    if task.is_classification:
        k = max(task.n_classes, 2)
        guess = math.log(k)
        out = [
            f"How to read: `logloss` is the average surprise at what actually happened (minus the log of the probability given to the real outcome). Lower is better. Guessing {k} ways equally scores {guess:.3f}; a perfect forecaster scores 0.",
            "`brier` is the average squared distance between the probabilities and the outcome; lower is better. `accuracy` is how often the most likely outcome was the actual one. Accuracy ignores how confident the forecast was, so logloss is the number to trust.",
            "`valid` rows were used to decide when to stop training. `test` rows were never seen while training and are the honest score.",
        ]
        if has_baseline:
            out.append("`baseline_logloss` is the same score for the comparison forecast (here, the bookmaker's de-vigged closing prices). `model_minus_baseline` is model minus baseline: negative means we beat it, positive means it beats us.")
        return out
    out = ["How to read: `rmse` is root mean squared error and `mae` mean absolute error, both in the target's units; lower is better."]
    if task.type == "poisson":
        out.append("`poisson_deviance` is the proper score for count targets; lower is better.")
    return out


def _curve_section(curve: pl.DataFrame, run_dir: Path) -> Section:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    folds = curve["fold"].unique(maintain_order=True).to_list()
    for i, f in enumerate(folds):
        part = curve.filter(pl.col("fold") == f)
        alpha = 1.0 if len(folds) == 1 else 0.35
        ax.plot(part["iteration"], part["train"], color=PALETTE[0], alpha=alpha, lw=1, label="train" if i == 0 else None)
        ax.plot(part["iteration"], part["valid"], color=PALETTE[1], alpha=alpha, lw=1, label="valid" if i == 0 else None)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("loss")
    ax.set_title("Learning curve" + (f" ({len(folds)} folds overlaid)" if len(folds) > 1 else ""))
    ax.legend()
    return Section("Learning curve", {}, [save_fig(fig, run_dir, "curve.png")], ["Valid loss should bottom out before the last round. If it is still falling, raise `rounds`; if the gap to train is wide, regularise."])


def _calibration_section(task: Task, pred: pl.DataFrame, run_dir: Path) -> Section:
    test = pred.filter(pl.col("split") == "test")
    table = calibration_table(task, test)
    balance = class_balance(task, test)
    names = task.class_names
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2), squeeze=False)
    for k, name in enumerate(names):
        ax = axes[0][k]
        part = table.filter(pl.col("class") == name)
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
        ax.scatter(part["mean_pred"], part["observed"], s=np.clip(part["n"].to_numpy() / 4, 8, 200), color=PALETTE[k % len(PALETTE)], alpha=0.8)
        ax.plot(part["mean_pred"], part["observed"], color=PALETTE[k % len(PALETTE)], lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"class {name}")
        ax.set_xlabel("predicted"); ax.set_ylabel("observed" if k == 0 else "")
    return Section("Calibration (test rows)", {"Class balance": balance, "Reliability": table}, [save_fig(fig, run_dir, "calibration.png")], ["Points on the diagonal mean the probabilities are honest. Bubble size is bin count."], {
        "Class balance": "For each outcome: how often it happened, the average probability the model gave it, and how often the model made it the favourite. If mean_pred matches observed_rate the model has the overall rates right.",
        "Reliability": "Rows are grouped by the probability the model gave a class. Within each group, `observed` is how often that class really happened. An honest model has observed close to mean_pred in every group; `gap` is the difference.",
    })


def _period_section(task: Task, pred: pl.DataFrame, period_col: str, baseline: list[str], run_dir: Path) -> Section:
    test = pred.filter(pl.col("split") == "test")
    table = period_table(task, test, period_col, baseline)
    figs: list[str] = []
    if task.is_classification and table.height:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(table.height)
        ax.bar(x - 0.2, table["logloss"], width=0.4, color=PALETTE[0], label="model")
        if "baseline_logloss" in table.columns:
            ax.bar(x + 0.2, table["baseline_logloss"].fill_null(np.nan), width=0.4, color=PALETTE[1], label="baseline")
        ax.set_xticks(x); ax.set_xticklabels(table["period"].to_list(), rotation=45, ha="right")
        lo = float(np.nanmin(table.select(pl.exclude("period", "n", "accuracy", "brier", "model_minus_baseline")).to_numpy().astype(float)))
        ax.set_ylim(max(0.0, lo - 0.05), None)
        ax.set_ylabel("logloss"); ax.set_title(f"Test logloss by {period_col}"); ax.legend()
        figs.append(save_fig(fig, run_dir, "periods.png"))
    return Section(f"By {period_col} (test rows)", {"Periods": table}, figs, [], {
        "Periods": f"The test scores split by {period_col}. Look for whether `model_minus_baseline` is consistently one sign or just bouncing around zero. A gap that only appears in one {period_col} is probably noise.",
    })


def _comparison_section(task: Task, pred: pl.DataFrame, compare: dict[str, list[str]], period_col: str | None, run_dir: Path) -> Section:
    test = pred.filter(pl.col("split") == "test")
    table = comparison_table(task, test, compare)
    tables = {"Comparisons": table}
    figs: list[str] = []
    if table.height:
        fig, ax = plt.subplots(figsize=(7, max(2.2, 0.45 * table.height + 0.8)))
        yy = np.arange(table.height)[::-1]
        d = table["diff"].to_numpy(); lo = table["diff_ci_lo"].to_numpy(); hi = table["diff_ci_hi"].to_numpy()
        ax.errorbar(d, yy, xerr=[d - lo, hi - d], fmt="o", color=PALETTE[0], ecolor=PALETTE[0], capsize=3)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_yticks(yy); ax.set_yticklabels([f"{f} (n {n})" for f, n in zip(table["forecast"], table["n"])])
        ax.set_xlabel("model logloss minus forecast logloss (left of 0: model better)"); ax.set_title("Model vs each comparison forecast, 95% intervals")
        figs.append(save_fig(fig, run_dir, "comparisons.png"))
    if period_col and period_col in test.columns:
        tables["By period"] = comparison_by_period(task, test, compare, period_col)
    return Section("Versus comparison forecasts (test rows)", tables, figs, [
        "Each comparison is like for like: only rows where that forecast exists, with its margin removed. The interval comes from the per-row loss differences, so it accounts for the two forecasts moving together.",
    ], {
        "Comparisons": "`diff` is model logloss minus forecast logloss on the shared rows: negative means the model is better. `diff_z` is diff over its standard error; only beyond about ±2 is the gap more than luck.",
        "By period": "`diff` per period for every comparison forecast. A real edge should be negative in most periods, not just one.",
    })


def _importance_section(importance: pl.DataFrame, run_dir: Path, top: int = 30) -> Section:
    head = importance.head(top)
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.22 * head.height)))
    ax.barh(head["feature"].to_list()[::-1], head["gain_share"].to_list()[::-1], color=PALETTE[0])
    ax.set_xlabel("share of total gain"); ax.set_title(f"Top {head.height} features by gain")
    return Section("Feature importance", {"Importance": head}, [save_fig(fig, run_dir, "importance.png")], ["Gain is how much each feature reduced the training loss across all splits. Averaged over folds."], {
        "Importance": "Which inputs the trees actually used. `gain_share` sums to 1 down the full list. Importance says what the model leaned on, not whether that input is any good out of sample; a feature can rank high and still hurt the test score.",
    })


def core_sections(project: Project, run_dir: Path, meta: dict[str, Any], pred: pl.DataFrame, task: Task) -> list[Section]:
    style()
    curve = pl.read_parquet(run_dir / "curve.parquet")
    importance = pl.read_parquet(run_dir / "importance.parquet")
    baseline = list(meta.get("baseline") or [])
    out = [_summary_section(meta, curve, task), _curve_section(curve, run_dir)]
    if task.is_classification:
        out.append(_calibration_section(task, pred, run_dir))
    if project.period_col and project.period_col in pred.columns:
        out.append(_period_section(task, pred, project.period_col, baseline, run_dir))
    compare = dict(meta.get("compare") or {})
    if compare and task.is_classification:
        out.append(_comparison_section(task, pred, compare, project.period_col, run_dir))
    out.append(_importance_section(importance, run_dir))
    return out


def hook_sections(project: Project, ctx: RunContext) -> list[Section]:
    if not project.has_module("evaluation"):
        return []
    mod = project.module("evaluation")
    if not hasattr(mod, "evaluate"):
        raise TypeError("evaluation module must define evaluate(ctx) -> list[Section]")
    sections = mod.evaluate(ctx)
    return list(sections or [])


# --- html --------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.4f}".rstrip("0").rstrip(".") if v != int(v) or abs(v) < 1 else f"{v:.0f}"
    return html.escape(str(v))


def _tip(text: str, glossary: dict[str, str]) -> str:
    """Wrap a heading or row label in a tooltip span when the glossary knows it."""
    key = str(text)
    definition = glossary.get(key) or glossary.get(key.lower())
    if definition is None:
        return html.escape(key)
    return f"<span class='tip' data-tip='{html.escape(definition, quote=True)}'>{html.escape(key)}</span>"


def table_html(df: pl.DataFrame, glossary: dict[str, str] | None = None) -> str:
    if df.height == 0:
        return "<p class='muted'>empty</p>"
    g = glossary or {}
    head = "".join(f"<th>{_tip(c, g)}</th>" for c in df.columns)
    rows = []
    for row in df.iter_rows():
        first, rest = row[0], row[1:]
        cells = [f"<td>{_tip(first, g) if isinstance(first, str) else _fmt(first)}</td>"] + [
            f"<td class='txt'>{_fmt(v)}</td>" if isinstance(v, str) and len(v) > 12 else f"<td>{_fmt(v)}</td>" for v in rest
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _terms_used(sections: list[Section], glossary: dict[str, str]) -> list[str]:
    seen: list[str] = []
    for s in sections:
        for df in s.tables.values():
            labels = list(df.columns) + ([str(v) for v in df[df.columns[0]].to_list() if isinstance(v, str)] if df.height else [])
            for lab in labels:
                key = lab if lab in glossary else (lab.lower() if lab.lower() in glossary else None)
                if key and key not in seen:
                    seen.append(key)
    return seen


def glossary_section(sections: list[Section], glossary: dict[str, str]) -> Section:
    terms = _terms_used(sections, glossary)
    df = pl.DataFrame({"term": terms, "meaning": [glossary[t] for t in terms]})
    return Section("Glossary", {"Terms on this page": df}, [], ["Every heading and row label above that has a definition is listed here, in the order it first appears."])


def _section_html(s: Section, glossary: dict[str, str] | None = None) -> str:
    parts = [f"<section><h2>{html.escape(s.title)}</h2>"]
    for n in s.notes:
        parts.append(f"<p class='note'>{_md_inline(n)}</p>")
    for f in s.figures:
        parts.append(f"<figure><img src='{html.escape(f)}' alt='{html.escape(f)}'></figure>")
    for name, df in s.tables.items():
        parts.append(f"<h3>{html.escape(name)}</h3>")
        if name in s.captions:
            parts.append(f"<p class='caption'>{_md_inline(s.captions[name])}</p>")
        parts.append(table_html(df, glossary if name != "Terms on this page" else None))
    parts.append("</section>")
    return "".join(parts)


def _md_inline(text: str) -> str:
    out = html.escape(text)
    while "`" in out:
        a = out.find("`"); b = out.find("`", a + 1)
        if b < 0:
            break
        out = out[:a] + "<code>" + out[a + 1 : b] + "</code>" + out[b + 1 :]
    return out


CSS = """
:root{--ink:#1a1f24;--muted:#5c6670;--line:#e4e7eb;--bg:#f6f7f8;--card:#fff;--accent:#1f4e79}
*{box-sizing:border-box}body{margin:0;font-family:"Segoe UI",system-ui,sans-serif;color:var(--ink);background:var(--bg);line-height:1.45}
header{background:var(--card);border-bottom:1px solid var(--line);padding:14px 24px;position:sticky;top:0;z-index:2;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
header h1{margin:0;font-size:17px}header a{color:var(--accent);text-decoration:none;font-size:13px}
main{max-width:1200px;margin:0 auto;padding:16px 24px 60px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin:16px 0}
h2{margin:0 0 8px;font-size:15px}h3{margin:14px 0 6px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
p.note{color:var(--muted);font-size:13px;margin:4px 0}code{background:#eef1f4;padding:1px 4px;border-radius:3px;font-size:12px}
table{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;margin:4px 0;display:block;overflow-x:auto;max-height:520px;overflow-y:auto}
th,td{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{background:#f0f2f4;position:sticky;top:0;text-align:right}
td:first-child,th:first-child{text-align:left}figure{margin:8px 0}img{max-width:100%;height:auto}
.muted{color:var(--muted)}nav{font-size:13px;margin:6px 0 0}nav a{margin-right:12px;color:var(--accent);text-decoration:none}
.runs td a{color:var(--accent);text-decoration:none}
p.caption{font-size:13px;color:var(--ink);margin:2px 0 8px;max-width:900px}
.tip{border-bottom:1px dotted var(--accent);cursor:help}
#tipbox{position:fixed;z-index:9;max-width:360px;background:#1a1f24;color:#fff;font-size:12.5px;line-height:1.4;padding:8px 10px;border-radius:6px;box-shadow:0 4px 14px rgba(0,0,0,.25);pointer-events:none;display:none;white-space:normal;text-align:left}
td.txt{text-align:left;white-space:normal;min-width:220px;max-width:560px;color:var(--muted)}
.runs td:nth-child(4){white-space:normal;min-width:360px;max-width:520px;text-align:left;font-size:12px;color:var(--muted)}
"""


TIP_JS = """
const box=document.createElement('div');box.id='tipbox';document.body.appendChild(box);
document.addEventListener('mouseover',e=>{const t=e.target.closest('.tip');if(!t)return;box.textContent=t.dataset.tip;box.style.display='block';});
document.addEventListener('mousemove',e=>{if(box.style.display!=='block')return;const w=box.offsetWidth,h=box.offsetHeight;let x=e.clientX+14,y=e.clientY+16;
if(x+w>window.innerWidth-8)x=e.clientX-w-14;if(y+h>window.innerHeight-8)y=e.clientY-h-12;box.style.left=x+'px';box.style.top=y+'px';});
document.addEventListener('mouseout',e=>{if(e.target.closest&&e.target.closest('.tip'))box.style.display='none';});
"""


def feature_docs(project: Project, run_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """(features, categorical, name -> description) for one run."""
    info = json.loads((run_dir / "features.json").read_text()) if (run_dir / "features.json").exists() else {}
    names = list(info.get("features") or [])
    cats = list(info.get("categorical") or [])
    docs: dict[str, str] = {}
    if project.has_module("features"):
        fn = getattr(project.module("features"), "describe", None)
        if callable(fn):
            for n in names:
                d = fn(n)
                if d:
                    docs[n] = str(d)
    return names, cats, docs


def features_section(names: list[str], cats: list[str], docs: dict[str, str], importance: pl.DataFrame) -> Section:
    share = dict(zip(importance["feature"].to_list(), importance["gain_share"].to_list()))
    df = pl.DataFrame({
        "feature": names,
        "type": ["category" if n in cats else "number" for n in names],
        "gain_share": [share.get(n) for n in names],
        "meaning": [docs.get(n, "") for n in names],
    })
    missing = [n for n in names if n not in docs]
    notes = [f"All {len(names)} inputs the model was allowed to see, in the order the project defines them. `gain_share` says how much the trees used each one (blank means never)."]
    if missing:
        notes.append(f"{len(missing)} features have no description yet: {', '.join(missing[:12])}{'…' if len(missing) > 12 else ''}.")
    return Section("Features used", {"Inputs": df}, [], notes)


def project_glossary(project: Project) -> dict[str, str]:
    g = dict(CORE_GLOSSARY)
    if project.has_module("evaluation"):
        g.update(getattr(project.module("evaluation"), "GLOSSARY", {}) or {})
    return g


def render_run(project: Project, run_dir: Path) -> Path:
    meta = load_meta(run_dir)
    pred = load_predictions(run_dir)
    task = Task.from_cfg(meta["task"]) if "classes" in meta["task"] else Task.from_cfg(meta["task"])
    sections = core_sections(project, run_dir, meta, pred, task)
    ctx = RunContext(project=project, model_cfg=meta.get("config") or {}, task=task, run_dir=run_dir, predictions=pred, meta=meta)
    sections += hook_sections(project, ctx)
    glossary = project_glossary(project)
    names, cats, docs = feature_docs(project, run_dir)
    sections.append(features_section(names, cats, docs, pl.read_parquet(run_dir / "importance.parquet")))
    sections.append(glossary_section(sections, glossary))
    tips = {**docs, **glossary}   # feature tooltips everywhere a feature name is a label; glossary wins on clashes
    nav = "".join(f"<a href='#s{i}'>{html.escape(s.title)}</a>" for i, s in enumerate(sections))
    body = "".join(_section_html(s, tips).replace("<section>", f"<section id='s{i}'>", 1) for i, s in enumerate(sections))
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(project.name)} / {html.escape(meta.get('model', ''))} / {html.escape(run_dir.name)}</title><style>{CSS}</style></head>
<body><header><h1>{html.escape(project.name)} / {html.escape(str(meta.get('model')))}</h1><span class="muted">{html.escape(run_dir.name)} · {html.escape(str(meta.get('created_at', ''))[:19])}</span><a href="../index.html">all runs</a></header>
<main><nav>{nav}</nav>{body}</main><script>{TIP_JS}</script></body></html>"""
    out = run_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    render_index(project)
    return out


def _yaml_description(project: Project, model: str) -> str:
    try:
        return str(project.model_cfg(model).get("description") or "")
    except FileNotFoundError:
        return ""


def render_index(project: Project) -> Path:
    rows = []
    for r in list_runs(project):
        m = load_meta(r)
        te = (m.get("metrics") or {}).get("test") or {}
        desc = m.get("description") or _yaml_description(project, str(m.get("model")))
        row: dict[str, Any] = {
            "run": f"<a href='{r.name}/index.html'>{html.escape(r.name)}</a>",
            "model": m.get("model"), "tag": m.get("tag") or "",
            "description": desc,
            "split": ((m.get("config") or {}).get("split") or {}).get("strategy", "holdout"),
            "rows": m.get("rows"), "features": m.get("n_features"),
            "test_n": te.get("n"),
        }
        if "logloss" in te:
            row.update({"test_logloss": te.get("logloss"), "baseline_logloss": te.get("baseline_logloss"), "model_minus_baseline": te.get("model_minus_baseline"), "diff_z": te.get("diff_z"), "seeds": m.get("seeds", 1), "seed_sd": te.get("seed_sd"), "accuracy": te.get("accuracy")})
        else:
            row.update({"test_rmse": te.get("rmse"), "test_mae": te.get("mae")})
        rows.append(row)
    rows.reverse()
    df = pl.DataFrame(rows) if rows else pl.DataFrame({"run": []})
    tbl = table_html(df, project_glossary(project)).replace("&lt;a href=&#x27;", "<a href='").replace("&#x27;&gt;", "'>").replace("&lt;/a&gt;", "</a>")
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(project.name)} runs</title><style>{CSS}</style></head>
<body><header><h1>{html.escape(project.name)} · runs</h1></header><main><section class="runs"><h2>Runs (newest first)</h2><p class="caption">One row per training run. `test_logloss` is the model's honest score, `baseline_logloss` the bookmaker's on the same rows, and `model_minus_baseline` the difference: negative means we beat the book. Hover a heading for its definition; click a run for its page.</p>{tbl}</section></main><script>{TIP_JS}</script></body></html>"""
    out = project.runs_dir / "index.html"
    project.runs_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out

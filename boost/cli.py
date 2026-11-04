from __future__ import annotations

import argparse
import sys

from boost.data import build_features, prepare
from boost.project import Project
from boost.report import render_index, render_run
from boost.run import train
from boost.store import list_runs, load_meta, resolve_run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="boost", description="gradient boosting runs for a project")
    parser.add_argument("--project", "-p", required=True, help="name under projects/ or a path")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="run the project's data recipe and cache prepared.parquet")
    f = sub.add_parser("features", help="build (or rebuild) the feature frame and print a summary")
    f.add_argument("--force", action="store_true")
    t = sub.add_parser("train", help="train a model config and write a run")
    t.add_argument("--model", "-m", required=True)
    t.add_argument("--tag", default=None, help="free text stored with the run")
    r = sub.add_parser("report", help="re-render a run page (re-runs evaluation hooks)")
    r.add_argument("--run", default="latest")
    sub.add_parser("runs", help="list runs")
    sub.add_parser("models", help="list model configs")
    args = parser.parse_args(argv)

    project = Project.find(args.project)
    if args.cmd == "prepare":
        out = prepare(project)
        print(f"prepared {out}")
        build_features(project, force=True)
        print(f"features {project.features_path}")
    elif args.cmd == "features":
        frame, names = build_features(project, force=args.force)
        print(f"{frame.height} rows, {len(names)} features -> {project.features_path}")
        for n in names:
            s = frame[n]
            print(f"  {n:34s} {str(s.dtype):12s} null {s.null_count() / frame.height:5.1%}")
    elif args.cmd == "train":
        run_dir = train(project, args.model, tag=args.tag)
        page = render_run(project, run_dir)
        print(f"page {page}")
    elif args.cmd == "report":
        run_dir = resolve_run(project, args.run)
        page = render_run(project, run_dir)
        print(f"page {page}")
    elif args.cmd == "runs":
        for r in list_runs(project):
            m = load_meta(r)
            te = (m.get("metrics") or {}).get("test") or {}
            key = "logloss" if "logloss" in te else "rmse"
            extra = f"  baseline {te['baseline_logloss']:.4f}  delta {te['model_minus_baseline']:+.4f}" if "baseline_logloss" in te else ""
            print(f"{r.name:40s} {m.get('model'):18s} test {key} {te.get(key, float('nan')):.4f}{extra}  {m.get('tag') or ''}")
        render_index(project)
    elif args.cmd == "models":
        for n in project.model_names():
            print(n)


if __name__ == "__main__":
    main(sys.argv[1:])

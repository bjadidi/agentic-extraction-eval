"""Command line entry point.

    python -m agex compare            # compare strategies, write reports/results.md + JSON
    python -m agex sweep              # threshold sweep for the cascade strategy
    python -m agex regress            # check the cascade against reports/baseline.json (exit 1 on regression)
    python -m agex run sds_007        # trace one document through the graph
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import ROOT, calibration_table, check_regression, evaluate, load_corpus
from .models import get_model
from .pipeline import Strategy, run_document

REPORTS = ROOT / "reports"


def strategies(small: str, large: str, threshold: float) -> list[Strategy]:
    s, l = get_model(small), get_model(large)
    return [
        Strategy("small model only", s, gate=False),
        Strategy("large model only", l, gate=False),
        Strategy("small + validation gate -> human", s, verifier=None, threshold=threshold),
        Strategy("cascade without grounding check", s, verifier=l, threshold=threshold, grounding=False),
        Strategy("cascade: small -> large -> human", s, verifier=l, threshold=threshold),
    ]


def cmd_compare(args) -> None:
    corpus = load_corpus()
    REPORTS.mkdir(exist_ok=True)
    rows, cascade_states, cascade_report = [], None, None
    for st in strategies(args.small, args.large, args.threshold):
        rep, states = evaluate(st, corpus)
        rows.append(rep.to_dict())
        if st.name.startswith("cascade"):
            cascade_states, cascade_report = states, rep
    (REPORTS / "results.json").write_text(json.dumps(rows, indent=2))
    calib = calibration_table(cascade_states, corpus)
    md = ["| Strategy | Auto-accepted accuracy | Silent errors | Automation | Straight-through docs | "
          "Final accuracy | Model $/doc | Total $/doc | p50 latency (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['strategy']} | {r['field_accuracy_auto']:.1%} | {r['silent_error_rate']:.2%} | "
                  f"{r['automation_rate']:.1%} | {r['doc_straight_through']:.0%} | {r['final_accuracy']:.1%} | "
                  f"{r['model_cost_per_doc']:.5f} | {r['total_cost_per_doc']:.3f} | {r['latency_p50_s']:.1f} |")
    md += ["", "Primary-model calibration (cascade run):", "", "| Confidence | n | Accuracy |", "|---|---|---|"]
    md += [f"| {c['bin']} | {c['n']} | {c['accuracy']:.1%} |" for c in calib]
    md += ["", "Per-layout final accuracy (cascade): " +
           ", ".join(f"layout {k}: {v:.1%}" for k, v in cascade_report.per_layout_accuracy.items())]
    (REPORTS / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def cmd_sweep(args) -> None:
    corpus = load_corpus()
    s, l = get_model(args.small), get_model(args.large)
    out = []
    print("| Threshold | Auto accuracy | Silent errors | Automation | Total $/doc |\n|---|---|---|---|---|")
    for t in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        rep, _ = evaluate(Strategy(f"cascade@{t}", s, verifier=l, threshold=t), corpus)
        out.append({"threshold": t, **rep.to_dict()})
        print(f"| {t:.2f} | {rep.field_accuracy_auto:.1%} | {rep.silent_error_rate:.2%} | "
              f"{rep.automation_rate:.1%} | {rep.total_cost_per_doc:.3f} |")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "threshold_sweep.json").write_text(json.dumps(out, indent=2))
    if args.plot:
        _plot_sweep(out)


def _plot_sweep(out: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = [o["automation_rate"] * 100 for o in out]
    y = [o["silent_error_rate"] * 100 for o in out]
    ax.plot(x, y, marker="o", color="#2b6cb0")
    labels: dict[tuple, list[str]] = {}
    for o, xi, yi in zip(out, x, y):
        labels.setdefault((round(xi, 3), round(yi, 3)), []).append(str(o["threshold"]))
    for (xi, yi), ts in labels.items():
        ax.annotate("t=" + "/".join(ts), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Fields auto-accepted (%)")
    ax.set_ylabel("Silent errors (% of all fields)")
    ax.set_title("Automation vs. silent-error trade-off (cascade)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "threshold_tradeoff.png", dpi=160)


def cmd_regress(args) -> None:
    corpus = load_corpus()
    st = strategies(args.small, args.large, args.threshold)[-1]
    rep, _ = evaluate(st, corpus)
    base_path = REPORTS / "baseline.json"
    if args.update or not base_path.exists():
        REPORTS.mkdir(exist_ok=True)
        base_path.write_text(json.dumps(rep.to_dict(), indent=2))
        print(f"baseline written to {base_path}")
        return
    failures = check_regression(rep.to_dict(), json.loads(base_path.read_text()))
    if failures:
        print("REGRESSION:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("no regression against baseline")


def cmd_run(args) -> None:
    text = (ROOT / "data" / "docs" / f"{args.doc_id}.txt").read_text()
    st = strategies(args.small, args.large, args.threshold)[-1]
    s = run_document(st, args.doc_id, text)
    print(f"path: {' -> '.join(s.trace)}")
    print(f"flagged: {s.flagged}\nescalated to human: {s.escalated}")
    for f, v in s.record.items():
        print(f"  {f:20s} {str(v):40s} conf={s.confidence.get(f, 0):.2f} source={s.source[f]}")
    print(f"cost ${s.cost_usd:.5f}, model latency {s.latency_s:.1f}s")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agex")
    ap.add_argument("--small", default="sim-small", help="primary model (sim-small or a Claude model id)")
    ap.add_argument("--large", default="sim-large", help="verifier model")
    ap.add_argument("--threshold", type=float, default=0.75)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compare")
    sw = sub.add_parser("sweep")
    sw.add_argument("--plot", action="store_true")
    rg = sub.add_parser("regress")
    rg.add_argument("--update", action="store_true", help="overwrite the stored baseline")
    rn = sub.add_parser("run")
    rn.add_argument("doc_id")
    args = ap.parse_args(argv)
    {"compare": cmd_compare, "sweep": cmd_sweep, "regress": cmd_regress, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()

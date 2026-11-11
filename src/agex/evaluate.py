"""Evaluation: field accuracy, automation rate, cost, latency, calibration, regressions."""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .pipeline import (HUMAN_REVIEW_S_PER_FIELD, HUMAN_REVIEW_USD_PER_FIELD, State, Strategy,
                       run_document)
from .schema import FIELD_NAMES, matches_gold

ROOT = Path(__file__).resolve().parents[2]


def load_corpus(root: Path = ROOT) -> list[tuple[dict, str]]:
    gold = [json.loads(line) for line in open(root / "data" / "gold.jsonl", encoding="utf-8")]
    return [(g, (root / "data" / "docs" / f"{g['doc_id']}.txt").read_text(encoding="utf-8")) for g in gold]


@dataclass
class Report:
    strategy: str
    n_docs: int
    field_accuracy_auto: float      # accuracy of fields the system accepted without a human
    silent_error_rate: float        # wrong fields that were auto-accepted / all fields
    automation_rate: float          # share of fields accepted without a human
    doc_straight_through: float     # docs with zero escalations
    final_accuracy: float           # after humans fix escalated fields (assumed correct)
    model_cost_per_doc: float
    total_cost_per_doc: float       # model + human review
    latency_p50_s: float
    latency_p95_s: float
    per_field_accuracy: dict
    per_layout_accuracy: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def evaluate(strategy: Strategy, corpus: list[tuple[dict, str]]) -> tuple[Report, list[State]]:
    states: list[State] = []
    auto_total = auto_correct = silent_wrong = 0
    per_field = {f: [0, 0] for f in FIELD_NAMES}
    per_layout: dict[int, list[int]] = {}
    straight = 0
    model_costs, total_costs, latencies = [], [], []
    for gold, text in corpus:
        s = run_document(strategy, gold["doc_id"], text)
        states.append(s)
        esc = set(s.escalated)
        straight += int(not esc)
        for f in FIELD_NAMES:
            ok = matches_gold(f, s.record.get(f), gold[f])
            per_field[f][1] += 1
            lay = per_layout.setdefault(gold["layout"], [0, 0])
            lay[1] += 1
            if f in esc:
                per_field[f][0] += 1  # human fixes it
                lay[0] += 1
                continue
            auto_total += 1
            auto_correct += int(ok)
            silent_wrong += int(not ok)
            per_field[f][0] += int(ok)
            lay[0] += int(ok)
        human_cost = len(esc) * HUMAN_REVIEW_USD_PER_FIELD
        model_costs.append(s.cost_usd)
        total_costs.append(s.cost_usd + human_cost)
        latencies.append(s.latency_s + len(esc) * HUMAN_REVIEW_S_PER_FIELD)
    n_fields = len(corpus) * len(FIELD_NAMES)
    rep = Report(
        strategy=strategy.name,
        n_docs=len(corpus),
        field_accuracy_auto=auto_correct / auto_total if auto_total else 0.0,
        silent_error_rate=silent_wrong / n_fields,
        automation_rate=auto_total / n_fields,
        doc_straight_through=straight / len(corpus),
        final_accuracy=(n_fields - silent_wrong) / n_fields,
        model_cost_per_doc=statistics.mean(model_costs),
        total_cost_per_doc=statistics.mean(total_costs),
        latency_p50_s=_pct(latencies, 0.5),
        latency_p95_s=_pct(latencies, 0.95),
        per_field_accuracy={f: c / t for f, (c, t) in per_field.items()},
        per_layout_accuracy={str(k): c / t for k, (c, t) in sorted(per_layout.items())},
    )
    return rep, states


def calibration_table(states: list[State], corpus: list[tuple[dict, str]], bins: int = 5) -> list[dict]:
    """Accuracy of the PRIMARY model's raw output by confidence bin (before routing)."""
    from .parsing import normalize

    rows = [[] for _ in range(bins)]
    for s, (gold, _) in zip(states, corpus):
        prim = normalize(s._primary.raw)
        for f in FIELD_NAMES:
            c = s._primary.confidence.get(f, 0.0)
            b = min(bins - 1, int(c * bins))
            rows[b].append(int(matches_gold(f, prim[f], gold[f])))
    out = []
    for i, r in enumerate(rows):
        if r:
            out.append({"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": len(r), "accuracy": sum(r) / len(r)})
    return out


REGRESSION_GATES = {
    "field_accuracy_auto": ("min", 0.005),   # may not drop more than 0.5 points
    "silent_error_rate": ("max", 0.002),     # may not rise more than 0.2 points
    "automation_rate": ("min", 0.03),
    "total_cost_per_doc": ("max", 0.05),     # relative tolerance for cost (5%)
}


def check_regression(current: dict, baseline: dict) -> list[str]:
    """Compare a report against a stored baseline; return a list of human-readable failures."""
    failures = []
    for metric, (direction, tol) in REGRESSION_GATES.items():
        cur, base = current[metric], baseline[metric]
        if metric == "total_cost_per_doc":
            if cur > base * (1 + tol):
                failures.append(f"{metric}: {cur:.4f} > baseline {base:.4f} (+{tol:.0%} allowed)")
        elif direction == "min" and cur < base - tol:
            failures.append(f"{metric}: {cur:.3f} < baseline {base:.3f} - {tol}")
        elif direction == "max" and cur > base + tol:
            failures.append(f"{metric}: {cur:.3f} > baseline {base:.3f} + {tol}")
    return failures

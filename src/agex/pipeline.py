"""The extraction graph.

    extract ─▶ normalize ─▶ validate ─▶ route ──(nothing flagged)──────────────▶ done
                                          │
                                          └─(flagged fields)─▶ verify ─▶ reconcile ─▶ done
                                                                 (2nd model,     (accept or
                                                                  flagged         escalate to
                                                                  fields only)    a human)

The graph is a small explicit state machine (`Graph`) so it has no framework
dependency; each node maps one-to-one onto a LangGraph `StateGraph` node if you
want to port it.

A field is *flagged* when the primary model's confidence is below `threshold` OR
it fails a deterministic validation rule. Flagged fields go to a stronger model
(only those fields, to keep cost down). A field is auto-accepted only if it
passes validation and the two models agree, or the stronger model is confident
on a field the first model got invalid. Everything else goes to a human queue.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .models import Extraction
from .parsing import normalize
from .schema import FIELD_NAMES, matches_gold, validate


def grounded(raw: str | None, text: str) -> bool:
    """True if the raw extracted string appears in the source as a whole span.

    Catches the confident errors that schema rules miss: a truncated name
    ("Granite Fin" inside "Granite Fine Chemicals") or a unit slip ("203 °C"
    when the document says "203 °F").
    """
    if raw is None:
        return True
    pat = r"(?<![\w.])" + re.escape(raw.strip()).replace(r"\ ", r"\s+") + r"(?![\w])"
    return re.search(pat, text, flags=re.IGNORECASE) is not None


HUMAN_REVIEW_USD_PER_FIELD = 0.40  # about 1 minute of a specialist's time per field
HUMAN_REVIEW_S_PER_FIELD = 60.0


@dataclass
class State:
    doc_id: str
    text: str
    record: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    valid: dict = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)
    escalated: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)  # field -> "primary" | "verifier" | "human"
    cost_usd: float = 0.0
    latency_s: float = 0.0
    trace: list[str] = field(default_factory=list)
    _primary: Extraction | None = None
    _verifier: Extraction | None = None


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Callable[[State], State]] = {}
        self.edges: dict[str, Callable[[State], str]] = {}

    def add_node(self, name: str, fn: Callable[[State], State]) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, router: Callable[[State], str] | str) -> None:
        self.edges[src] = router if callable(router) else (lambda s, d=router: d)

    def run(self, state: State, start: str) -> State:
        node = start
        while node != "END":
            state.trace.append(node)
            state = self.nodes[node](state)
            node = self.edges[node](state)
        return state


@dataclass
class Strategy:
    """How to run the graph.

    primary:   model used for the first full extraction
    verifier:  stronger model for flagged fields (None = skip straight to humans)
    threshold: confidence below which a field is flagged
    gate:      if False, accept the primary output as-is (no validation routing)
    grounding: require every extracted value to appear verbatim in the document
    """

    name: str
    primary: object
    verifier: object | None = None
    threshold: float = 0.75
    gate: bool = True
    grounding: bool = True


def build_graph(strategy: Strategy) -> Graph:
    g = Graph()

    def extract(s: State) -> State:
        ex = strategy.primary.extract(s.doc_id, s.text)
        s._primary = ex
        s.cost_usd += ex.cost_usd
        s.latency_s += ex.latency_s
        s.confidence = dict(ex.confidence)
        s.record = normalize(ex.raw)
        s.source = {f: "primary" for f in FIELD_NAMES}
        return s

    def validate_node(s: State) -> State:
        s.valid = validate(s.record)
        if strategy.grounding:
            for f in FIELD_NAMES:
                s.valid[f] = s.valid[f] and grounded(s._primary.raw.get(f), s.text)
        return s

    def route(s: State) -> State:
        if not strategy.gate:
            s.flagged = []
            return s
        s.flagged = [f for f in FIELD_NAMES
                     if not s.valid[f] or s.confidence.get(f, 0.0) < strategy.threshold]
        return s

    def verify(s: State) -> State:
        ex = strategy.verifier.extract(s.doc_id, s.text, fields=s.flagged)
        s._verifier = ex
        s.cost_usd += ex.cost_usd
        s.latency_s += ex.latency_s
        return s

    def reconcile(s: State) -> State:
        ver = s._verifier
        ver_norm = normalize({**s._primary.raw, **(ver.raw if ver else {})}) if ver else {}
        for f in s.flagged:
            if ver is None:
                s.escalated.append(f)
                continue
            candidate = {**s.record, f: ver_norm[f]}
            ok = validate(candidate)[f] and (not strategy.grounding or grounded(ver.raw.get(f), s.text))
            agree = matches_gold(f, ver_norm[f], s.record[f])
            confident = ver.confidence.get(f, 0.0) >= strategy.threshold
            if ok and (agree or (not s.valid[f] and confident)):
                s.record[f] = ver_norm[f]
                s.source[f] = "verifier"
                s.confidence[f] = ver.confidence.get(f, 0.0)
            else:
                s.escalated.append(f)
        return s

    g.add_node("extract", extract)
    g.add_node("normalize_validate", validate_node)
    g.add_node("route", route)
    g.add_node("verify", verify)
    g.add_node("reconcile", reconcile)
    g.add_edge("extract", "normalize_validate")
    g.add_edge("normalize_validate", "route")
    g.add_edge("route", lambda s: "END" if not s.flagged else ("verify" if strategy.verifier else "reconcile"))
    g.add_edge("verify", "reconcile")
    g.add_edge("reconcile", "END")
    return g


def run_document(strategy: Strategy, doc_id: str, text: str) -> State:
    return build_graph(strategy).run(State(doc_id=doc_id, text=text), start="extract")

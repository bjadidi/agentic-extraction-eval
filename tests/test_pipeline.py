import json
from pathlib import Path

from agex.evaluate import check_regression, evaluate, load_corpus
from agex.models import SimulatedModel
from agex.pipeline import Strategy, grounded, run_document

ROOT = Path(__file__).resolve().parents[1]
CORPUS = load_corpus(ROOT)


def test_grounding_catches_truncation_and_unit_slip():
    text = "Supplier: Granite Fine Chemicals\nBoiling point: 132.8 °F\n"
    assert grounded("Granite Fine Chemicals", text)
    assert not grounded("Granite Fin", text)
    assert not grounded("132.8 °C", text)
    assert grounded("132.8 °F", text)


def test_simulation_is_deterministic():
    m = SimulatedModel("sim-small")
    g, text = CORPUS[5]
    a, b = m.extract(g["doc_id"], text), m.extract(g["doc_id"], text)
    assert a.raw == b.raw and a.confidence == b.confidence


def test_unflagged_document_skips_verifier():
    st = Strategy("t", SimulatedModel("sim-small"), verifier=SimulatedModel("sim-large"), threshold=0.0,
                  grounding=False)
    g, text = CORPUS[0]
    s = run_document(st, g["doc_id"], text)
    if not s.flagged:
        assert "verify" not in s.trace


def test_cascade_beats_small_model_on_silent_errors():
    small = SimulatedModel("sim-small")
    large = SimulatedModel("sim-large")
    r_small, _ = evaluate(Strategy("small", small, gate=False), CORPUS)
    r_cascade, _ = evaluate(Strategy("cascade", small, verifier=large), CORPUS)
    assert r_cascade.silent_error_rate < r_small.silent_error_rate / 3
    assert r_cascade.model_cost_per_doc < 0.5 * evaluate(Strategy("large", large, gate=False), CORPUS)[0].model_cost_per_doc


def test_grounding_reduces_silent_errors():
    small, large = SimulatedModel("sim-small"), SimulatedModel("sim-large")
    with_g, _ = evaluate(Strategy("g", small, verifier=large), CORPUS)
    without, _ = evaluate(Strategy("ng", small, verifier=large, grounding=False), CORPUS)
    assert with_g.silent_error_rate < without.silent_error_rate


def test_regression_gate_flags_a_drop():
    base = json.loads((ROOT / "reports" / "baseline.json").read_text())
    worse = {**base, "field_accuracy_auto": base["field_accuracy_auto"] - 0.02}
    assert check_regression(worse, base)
    assert not check_regression(base, base)

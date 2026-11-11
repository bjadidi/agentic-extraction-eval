import json
from pathlib import Path

import pytest

from agex.parsing import normalize, parse_temperature_c, read_fields
from agex.schema import FIELD_NAMES, cas_checksum_ok, matches_gold, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cas,ok", [("67-64-1", True), ("64-17-5", True), ("7732-18-5", True),
                                    ("76-64-1", False), ("67-64-2", False), ("6764-1", False)])
def test_cas_checksum(cas, ok):
    assert cas_checksum_ok(cas) is ok


@pytest.mark.parametrize("raw,expected", [("56 °C", 56.0), ("132.8 °F", 56.0), ("Not applicable", None),
                                          ("-4 °C", -4.0), (None, None)])
def test_temperature_normalization(raw, expected):
    assert parse_temperature_c(raw) == expected


def test_reference_parser_matches_gold_on_every_layout():
    gold = [json.loads(line) for line in open(ROOT / "data" / "gold.jsonl")]
    for g in gold:
        rec = normalize(read_fields((ROOT / "data" / "docs" / f"{g['doc_id']}.txt").read_text()))
        for f in FIELD_NAMES:
            assert matches_gold(f, rec[f], g[f]), (g["doc_id"], f, rec[f], g[f])
        assert all(validate(rec).values()), g["doc_id"]


def test_validation_rejects_flash_point_above_boiling_point():
    rec = {"product_name": "X", "manufacturer": "Y Co", "cas_number": "67-64-1", "ghs_classification": "Flammable",
           "signal_word": "Danger", "flash_point_c": 120.0, "boiling_point_c": 56.0, "ph": None}
    assert validate(rec)["flash_point_c"] is False

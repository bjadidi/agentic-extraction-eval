"""Layout-agnostic reading of SDS text plus unit normalization.

`read_fields` returns raw strings exactly as written in the document (e.g. "203.0 °F").
`normalize` turns raw strings into typed, canonical values (°C floats, None for
"not applicable", canonical signal words). Keeping the two steps separate lets the
pipeline catch a model that reads the right number but forgets the unit.
"""
from __future__ import annotations

import re

_PATTERNS: dict[str, list[str]] = {
    "product_name": [
        r"Product identifier:\s*(.+)",
        r"Trade name \.+\s*(.+)",
        r"This document covers (.+?) \(CAS",
        r"^sds // (.+)$",
    ],
    "manufacturer": [
        r"Supplier:\s*(.+)",
        r"^(.+?) -- PRODUCT SAFETY INFORMATION",
        r"prepared by (.+?)\. Rev",
        r"manufacturer:\s*(.+)",
    ],
    "cas_number": [r"\bCAS(?: No\.:| registry \.+| |#)\s*(\d{2,7}-\d{2}-\d)"],
    "ghs_classification": [
        r"GHS classification:\s*(.+)",
        r"Classification:\s*(.+?); signal word",
        r"classified as (.+?), and ",
        r"ghs:\s*(.+)",
    ],
    "signal_word": [
        r"Signal word:\s*(\w+)",
        r"signal word = (\w+)",
        r"the signal word is '(\w+)'",
        r"(no signal word) is required",
        r"signal word:\s*(\w+)",
    ],
    "flash_point_c": [
        r"Flash point:\s*([^\n]+)",
        r"\| Flash point \(closed cup\) \| ([^|]+) \|",
        r"flash point is (-?[\d.]+\s*°[CF])",
        r"(A flash point is not applicable)",
        r"fp:\s*([^\n]+)",
    ],
    "boiling_point_c": [
        r"Boiling point:\s*([^\n]+)",
        r"\| Boiling point / range \| ([^|]+) \|",
        r"boils at (-?[\d.]+\s*°[CF])",
        r"bp:\s*([^\n]+)",
    ],
    "ph": [
        r"\bpH:\s*([^\n]+)",
        r"\| pH \(as supplied\) \| ([^|]+) \|",
        r"pH of the supplied product is approximately ([\d.]+)",
        r"\bph:\s*([^\n]+)",
    ],
}


def read_fields(text: str) -> dict[str, str | None]:
    """Return raw string per field (None if no pattern matched)."""
    out: dict[str, str | None] = {}
    for field, pats in _PATTERNS.items():
        val = None
        for p in pats:
            m = re.search(p, text, flags=re.MULTILINE | (re.IGNORECASE if field in ("ph", "cas_number") else 0))
            if m:
                val = m.group(1).strip().rstrip(".")
                break
        out[field] = val
    return out


_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*([CF])", re.IGNORECASE)
_NA = re.compile(r"not (applicable|available)|^n/?a$", re.IGNORECASE)


def parse_temperature_c(raw: str | None) -> float | None:
    if raw is None or _NA.search(raw):
        return None
    m = _TEMP.search(raw)
    if not m:
        try:
            return float(raw)
        except ValueError:
            return None
    v, unit = float(m.group(1)), m.group(2).upper()
    return round((v - 32) * 5 / 9, 1) if unit == "F" else v


def parse_number(raw: str | None) -> float | None:
    if raw is None or _NA.search(raw):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    return float(m.group(0)) if m else None


def normalize_signal(raw: str | None) -> str | None:
    if raw is None:
        return None
    r = raw.strip().lower()
    if r in {"none", "no signal word", "n/a"}:
        return "None"
    return r.capitalize() if r in {"danger", "warning"} else raw.strip()


def normalize(raw: dict[str, str | None]) -> dict:
    return {
        "product_name": raw.get("product_name"),
        "manufacturer": (raw.get("manufacturer") or "").title() or None,
        "cas_number": raw.get("cas_number"),
        "ghs_classification": raw.get("ghs_classification"),
        "signal_word": normalize_signal(raw.get("signal_word")),
        "flash_point_c": parse_temperature_c(raw.get("flash_point_c")),
        "boiling_point_c": parse_temperature_c(raw.get("boiling_point_c")),
        "ph": parse_number(raw.get("ph")),
    }

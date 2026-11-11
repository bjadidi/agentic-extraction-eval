"""Model adapters.

`SimulatedModel` is a deterministic stand-in for an LLM so the whole pipeline,
its evaluation and its regression tests run offline and reproducibly. It reads
the document with a layout-agnostic parser and then injects the error types we
see from real models, at rates that depend on the model tier and on how hard the
layout is:

* miss        - field not returned
* unit_slip   - reads "203 °F" but reports 203 as Celsius (silent, often confident)
* digit_swap  - transposes two digits in a CAS number (caught by the checksum)
* truncation  - cuts a long name short

Confidence scores are imperfectly calibrated on purpose: about half of the
injected errors come back with high confidence, which is exactly why the
pipeline does not trust confidence alone.

`AnthropicModel` calls a real model through the Anthropic SDK with the same
interface, so you can swap it in and re-run the same evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field

from .parsing import read_fields
from .schema import FIELD_NAMES

TEMP_FIELDS = {"flash_point_c", "boiling_point_c"}


def _u(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from string parts."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


@dataclass
class Extraction:
    raw: dict[str, str | None]
    confidence: dict[str, float]
    cost_usd: float
    latency_s: float
    model: str
    injected: dict[str, str] = field(default_factory=dict)  # simulation only: field -> error type


@dataclass
class Tier:
    name: str
    base_error: float
    hard_layout_multiplier: float
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    base_latency_s: float


TIERS = {
    "sim-small": Tier("sim-small", 0.09, 2.5, 0.25, 1.25, 0.6),
    "sim-large": Tier("sim-large", 0.02, 1.5, 3.00, 15.0, 2.2),
}


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class SimulatedModel:
    def __init__(self, tier: str = "sim-small", seed: str = "0") -> None:
        self.tier = TIERS[tier]
        self.name = tier
        self.seed = seed

    def extract(self, doc_id: str, text: str, fields: list[str] | None = None) -> Extraction:
        fields = fields or FIELD_NAMES
        truth = read_fields(text)
        hard = "Physical data:" in text  # narrative layout
        p_err = self.tier.base_error * (self.tier.hard_layout_multiplier if hard else 1.0)
        raw, conf, injected = {}, {}, {}
        for f in fields:
            value = truth.get(f)
            u = _u(self.seed, self.name, doc_id, f)
            if value is not None and u < p_err:
                kind = self._error_kind(f, value, _u(self.seed, self.name, doc_id, f, "kind"))
                value = self._apply(kind, value)
                injected[f] = kind
                overconfident = _u(self.seed, self.name, doc_id, f, "oc") < 0.5
                conf[f] = 0.80 + 0.17 * _u(doc_id, f, "c1") if overconfident else 0.30 + 0.40 * _u(doc_id, f, "c2")
            else:
                conf[f] = 0.78 + 0.21 * _u(self.name, doc_id, f, "c3")
            raw[f] = value
        out_tokens = 20 + 15 * len(fields)
        cost = (_tokens(text) * self.tier.usd_per_mtok_in + out_tokens * self.tier.usd_per_mtok_out) / 1e6
        latency = self.tier.base_latency_s + out_tokens * (0.004 if "small" in self.name else 0.012)
        return Extraction(raw, conf, cost, latency, self.name, injected)

    @staticmethod
    def _error_kind(f: str, value: str, u: float) -> str:
        if f in TEMP_FIELDS and "°F" in value:
            return "unit_slip" if u < 0.7 else "miss"
        if f == "cas_number":
            return "digit_swap" if u < 0.7 else "miss"
        if f in {"product_name", "manufacturer", "ghs_classification"} and len(value) > 12:
            return "truncation" if u < 0.6 else "miss"
        return "miss"

    @staticmethod
    def _apply(kind: str, value: str) -> str | None:
        if kind == "miss":
            return None
        if kind == "unit_slip":
            return re.sub(r"°\s*F", "°C", value)
        if kind == "digit_swap":
            d = list(value)
            idx = [i for i, ch in enumerate(d) if ch.isdigit()]
            i, j = idx[0], idx[1]
            if d[i] == d[j] and len(idx) > 2:
                j = idx[2]
            d[i], d[j] = d[j], d[i]
            return "".join(d)
        if kind == "truncation":
            return value[: max(4, len(value) // 2)].rstrip()
        return value


PROMPT = """You extract fields from a chemical Safety Data Sheet (SDS).
Return ONLY a JSON object with these keys: {keys}.
Each value must be an object {{"value": <string exactly as written in the document, or null>, "confidence": <0..1>}}.
Copy numbers WITH their units as written (e.g. "203.0 °F"). Do not convert units. Use null if absent.

Document:
<<<
{doc}
>>>"""


class AnthropicModel:
    """Real-model adapter (requires `pip install anthropic` and ANTHROPIC_API_KEY)."""

    PRICES = {  # USD per million tokens (input, output); update to current pricing
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-sonnet-4-5": (3.0, 15.0),
    }

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        import anthropic  # imported lazily so the package works offline

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.name = model

    def extract(self, doc_id: str, text: str, fields: list[str] | None = None) -> Extraction:
        fields = fields or FIELD_NAMES
        t0 = time.perf_counter()
        msg = self.client.messages.create(
            model=self.name,
            max_tokens=800,
            messages=[{"role": "user", "content": PROMPT.format(keys=", ".join(fields), doc=text)}],
        )
        latency = time.perf_counter() - t0
        body = msg.content[0].text
        data = json.loads(body[body.find("{"): body.rfind("}") + 1])
        raw = {f: (data.get(f) or {}).get("value") for f in fields}
        conf = {f: float((data.get(f) or {}).get("confidence") or 0.0) for f in fields}
        p_in, p_out = self.PRICES.get(self.name, (3.0, 15.0))
        cost = (msg.usage.input_tokens * p_in + msg.usage.output_tokens * p_out) / 1e6
        return Extraction(raw, conf, cost, latency, self.name)


def get_model(name: str, seed: str = "0"):
    if name in TIERS:
        return SimulatedModel(name, seed=seed)
    return AnthropicModel(name)

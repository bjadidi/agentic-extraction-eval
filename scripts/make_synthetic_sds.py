"""Generate a synthetic corpus of chemical Safety Data Sheet (SDS) excerpts with gold labels.

The documents are deliberately non-uniform: four different layouts, mixed units
(Celsius / Fahrenheit), optional fields, and noisy formatting. This mimics the
kind of long, inconsistent technical documents that make extraction hard.

Usage:
    python scripts/make_synthetic_sds.py --n 60 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUBSTANCES = [
    ("Acetone", "67-64-1", -20, 56, "Flammable liquid, Category 2", "Danger", None),
    ("Ethanol", "64-17-5", 13, 78, "Flammable liquid, Category 2", "Danger", None),
    ("Isopropanol", "67-63-0", 12, 82, "Flammable liquid, Category 2", "Danger", None),
    ("Toluene", "108-88-3", 4, 111, "Flammable liquid, Category 2", "Danger", None),
    ("Hexane", "110-54-3", -22, 69, "Flammable liquid, Category 2", "Danger", None),
    ("Ethyl acetate", "141-78-6", -4, 77, "Flammable liquid, Category 2", "Danger", None),
    ("Methanol", "67-56-1", 11, 65, "Flammable liquid, Category 2", "Danger", None),
    ("Glycerol", "56-81-5", 160, 290, "Not classified", "None", None),
    ("Propylene glycol", "57-55-6", 99, 188, "Not classified", "None", None),
    ("Sodium hydroxide solution 50%", "1310-73-2", None, 140, "Skin corrosion, Category 1A", "Danger", 14.0),
    ("Hydrochloric acid 37%", "7647-01-0", None, 57, "Skin corrosion, Category 1B", "Danger", 0.5),
    ("Acetic acid, glacial", "64-19-7", 39, 118, "Flammable liquid, Category 3", "Danger", 2.4),
    ("Citric acid solution 10%", "77-92-9", None, 100, "Eye irritation, Category 2", "Warning", 2.2),
    ("Hydrogen peroxide 30%", "7722-84-1", None, 106, "Oxidizing liquid, Category 2", "Danger", 3.5),
    ("Sodium hypochlorite 12%", "7681-52-9", None, 111, "Skin corrosion, Category 1B", "Danger", 12.5),
    ("Ethylene glycol", "107-21-1", 111, 197, "Acute toxicity (oral), Category 4", "Warning", None),
    ("Xylene (mixed isomers)", "1330-20-7", 27, 140, "Flammable liquid, Category 3", "Warning", None),
    ("n-Butanol", "71-36-3", 35, 118, "Flammable liquid, Category 3", "Warning", None),
    ("Dimethyl sulfoxide", "67-68-5", 87, 189, "Not classified", "None", None),
    ("Mineral oil", "8042-47-5", 185, 300, "Aspiration hazard, Category 1", "Danger", None),
]
MANUFACTURERS = ["Northwind Chemicals", "Lakeside Reagents", "Polaris Lab Supply", "Granite Fine Chemicals",
                 "Harbor Industrial", "Maple Leaf Solvents"]


def c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def render(sub, mfr, layout, rng):
    name, cas, fp, bp, hazard, signal, ph = sub
    use_f = rng.random() < 0.3
    fp_txt = "Not applicable" if fp is None else (f"{c_to_f(fp)} °F" if use_f else f"{fp} °C")
    bp_txt = f"{c_to_f(bp)} °F" if use_f else f"{bp} °C"
    ph_txt = "Not available" if ph is None else f"{ph}"
    rev = f"Rev. {rng.randint(1, 9)}.{rng.randint(0, 9)} ({rng.randint(2019, 2025)})"
    filler = rng.choice([
        "Keep container tightly closed. Store in a cool, well-ventilated place.",
        "Wear protective gloves, eye protection and face protection.",
        "In case of fire: use dry chemical, CO2 or alcohol-resistant foam.",
        "Avoid release to the environment. Dispose of contents in accordance with local regulations.",
    ])
    if layout == 0:
        return (f"SAFETY DATA SHEET\nSection 1: Identification\nProduct identifier: {name}\nSupplier: {mfr}\n"
                f"CAS No.: {cas}\n\nSection 2: Hazard(s) identification\nGHS classification: {hazard}\n"
                f"Signal word: {signal}\n{filler}\n\nSection 9: Physical and chemical properties\n"
                f"Flash point: {fp_txt}\nBoiling point: {bp_txt}\npH: {ph_txt}\n\n{rev}\n")
    if layout == 1:
        return (f"{mfr.upper()} -- PRODUCT SAFETY INFORMATION\n{rev}\n\n1. PRODUCT\n   Trade name ........ {name}\n"
                f"   CAS registry ...... {cas}\n2. HAZARDS\n   Classification: {hazard}; signal word = {signal}\n"
                f"   Precautions: {filler}\n9. PROPERTIES\n   | Property | Value |\n   | Boiling point / range | {bp_txt} |\n"
                f"   | Flash point (closed cup) | {fp_txt} |\n   | pH (as supplied) | {ph_txt} |\n")
    if layout == 2:
        # narrative layout, fields buried in prose
        sig = "no signal word is required" if signal == "None" else f"the signal word is '{signal}'"
        fp_sent = ("A flash point is not applicable." if fp is None else f"Its closed-cup flash point is {fp_txt}.")
        ph_sent = "" if ph is None else f" The pH of the supplied product is approximately {ph_txt}."
        return (f"Product safety summary prepared by {mfr}. {rev}.\n\nThis document covers {name} "
                f"(CAS {cas}). Under GHS the product is classified as {hazard}, and {sig}. {filler}\n\n"
                f"Physical data: the substance boils at {bp_txt}. {fp_sent}{ph_sent}\n")
    # layout 3: key/value with odd spacing and lowercase keys
    return (f"sds // {name}\nmanufacturer:{mfr}\ncas#   {cas}\n\nhazards\n  ghs: {hazard}\n  signal word:   {signal}\n\n"
            f"phys/chem\n  bp: {bp_txt}\n  fp: {fp_txt}\n  ph: {ph_txt}\n\nnotes: {filler}\n{rev}\n")


def gold(sub, mfr):
    name, cas, fp, bp, hazard, signal, ph = sub
    return {
        "product_name": name,
        "manufacturer": mfr,
        "cas_number": cas,
        "ghs_classification": hazard,
        "signal_word": signal,
        "flash_point_c": None if fp is None else float(fp),
        "boiling_point_c": float(bp),
        "ph": ph,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    docs_dir = ROOT / "data" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for f in docs_dir.glob("*.txt"):
        f.unlink()
    rows = []
    for i in range(args.n):
        sub = rng.choice(SUBSTANCES)
        mfr = rng.choice(MANUFACTURERS)
        layout = i % 4
        doc_id = f"sds_{i:03d}"
        (docs_dir / f"{doc_id}.txt").write_text(render(sub, mfr, layout, rng), encoding="utf-8")
        rows.append({"doc_id": doc_id, "layout": layout, **gold(sub, mfr)})
    with open(ROOT / "data" / "gold.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} documents to {docs_dir}")


if __name__ == "__main__":
    main()

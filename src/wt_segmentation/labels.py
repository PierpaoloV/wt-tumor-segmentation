"""
Canonical class dictionary for the Wilms' tumour segmentation project.

All scripts (rasterisation, split generation, config generation, evaluation)
import from here so the Dutch→English→mask_int mapping never drifts.

Mask integer scheme
-------------------
Integers 0–15 are assigned to the 16 classes below.
Fibrose (Dutch: Fibrose) is merged into Regressie at rasterisation time by
mapping both Dutch group names to the same integer (6).

Dropped classes (not rasterised, no entry here):
  Anaplasie          — 0 annotated polygons
  Bijnierschors      — not relevant to WT diagnosis
  Bijniermerg        — not relevant to WT diagnosis
  Normaal nierweefsel — 1 polygon; excluded per thesis methodology

EMPTY_VALUE (255)
-----------------
Unannotated pixels in sparse masks receive this value.  It is intentionally
outside {0..15} so the pipeline sampler auto-ignores those pixels
(set to ignore_index=-100 in torch_data_generator.py:117).
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Core definition
# Each entry: (dutch_xml_group_name, english_name, mask_int)
# ---------------------------------------------------------------------------
_CLASSES: List[Tuple[str, str, int]] = [
    ("Achtergrond",    "background",          0),
    ("WT-blasteem",    "wt_blastema",         1),
    ("WT-stroma",      "wt_stroma",           2),
    ("WT-epitheel",    "wt_epithelium",       3),
    ("Necrose",        "necrosis",            4),
    ("Bloeding",       "bleeding",            5),
    ("Regressie",      "regression",          6),
    ("Fibrose",        "regression",          6),   # merged into Regressie
    ("Glomeruli",      "glomeruli",           7),
    ("Tubuli",         "tubules",             8),
    ("Nefrogene rest", "nephrogenic_rest",    9),
    ("Vet",            "fat",                10),
    ("Bindweefsel",    "connective_tissue",  11),
    ("Bloedvaten",     "blood_vessels",      12),
    ("Zenuwtakken",    "nerves",             13),
    ("Lymfklier",      "lymph_nodes",        14),
    ("Urotheel",       "urothelium",         15),
]

EMPTY_VALUE: int = 255  # pixel value for unannotated regions

# ---------------------------------------------------------------------------
# Derived look-ups
# ---------------------------------------------------------------------------

# Dutch XML group name → mask integer  (Fibrose→6, same as Regressie)
DUTCH_TO_INT: Dict[str, int] = {dutch: idx for dutch, _, idx in _CLASSES}

# Mask integer → English name  (first entry wins for int 6 → "regression")
INT_TO_ENGLISH: Dict[int, str] = {}
for _, eng, idx in _CLASSES:
    INT_TO_ENGLISH.setdefault(idx, eng)

# English name → mask integer  (unique, Fibrose already merged)
ENGLISH_TO_INT: Dict[str, int] = {eng: idx for _, eng, idx in _CLASSES}

# Sorted list of unique mask integers (0–15)
ALL_INTS: List[int] = sorted(set(idx for _, _, idx in _CLASSES))

NUM_CLASSES: int = len(ALL_INTS)   # 16

# ---------------------------------------------------------------------------
# Rasterisation overlay order
# Groups listed LAST are painted on top (highest priority).
# WT-* and pathological classes win over background and normal stroma.
# ---------------------------------------------------------------------------
ORDER: List[str] = [
    "Achtergrond",
    "Vet",
    "Bindweefsel",
    "Bloedvaten",
    "Zenuwtakken",
    "Lymfklier",
    "Urotheel",
    "Tubuli",
    "Glomeruli",
    "Nefrogene rest",
    "WT-stroma",
    "WT-epitheel",
    "Regressie",
    "Fibrose",        # painted with same int as Regressie → merge
    "Bloeding",
    "Necrose",
    "WT-blasteem",
]

# ---------------------------------------------------------------------------
# Sampling weights (label_dist for network_configuration.yaml)
#
# Interim values derived from  1/sqrt(polygon_count)  normalised to sum=1.
# Polygon counts from PROJECT_REPORT.md §2.2.  Fibrose (16) is folded into
# Regressie (484) → 500 combined.  Background is manually overridden to
# 0.020 because polygon count (90) massively underestimates pixel-area
# prevalence.
#
# IMPORTANT: regenerate from pixel histograms after first rasterisation
# (scripts/make_splits.py --recompute-weights).
# ---------------------------------------------------------------------------
LABEL_DIST: Dict[int, float] = {
    0:  0.0200,   # background          (90 polygons — manual override)
    1:  0.0494,   # wt_blastema         (441)
    2:  0.0557,   # wt_stroma           (349)
    3:  0.0392,   # wt_epithelium       (704)
    4:  0.0688,   # necrosis            (229)
    5:  0.0656,   # bleeding            (251)
    6:  0.0465,   # regression+fibrosis (500)
    7:  0.0440,   # glomeruli           (557)
    8:  0.0458,   # tubules             (515)
    9:  0.0847,   # nephrogenic_rest    (151)
    10: 0.0603,   # fat                 (298)
    11: 0.0441,   # connective_tissue   (556)
    12: 0.0339,   # blood_vessels       (944)
    13: 0.0838,   # nerves              (154)
    14: 0.1365,   # lymph_nodes         (58)
    15: 0.1217,   # urothelium          (73)
}

# Identity label_map: mask int == class index (0..15 are already contiguous)
LABEL_MAP: Dict[int, int] = {i: i for i in ALL_INTS}


def recompute_label_dist(pixel_counts: Dict[int, int]) -> Dict[int, float]:
    """Recompute label_dist from actual pixel counts using 1/sqrt(count).

    Args:
        pixel_counts: {mask_int: pixel_count} from histogram over training masks.
                      Mask ints absent (zero pixels) are excluded.

    Returns:
        Normalised {mask_int: weight} summing to 1.0.
    """
    import math

    raw = {k: 1.0 / math.sqrt(v) for k, v in pixel_counts.items() if v > 0}
    total = sum(raw.values())
    return {k: round(v / total, 6) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# CLI --dump for use inside evaluate.sh
# ---------------------------------------------------------------------------
_DUMP_FORMATS = ("awesomedice_classes", "identity_map", "label_dist", "all")


def _dump(fmt: str) -> str:
    if fmt == "awesomedice_classes":
        # {english_name: mask_int} for awesomedice.py -c flag
        return json.dumps({eng: idx for idx, eng in INT_TO_ENGLISH.items()})
    if fmt == "identity_map":
        # {mask_int: class_idx} for awesomedice.py -m flag (identity here)
        return json.dumps({str(k): v for k, v in LABEL_MAP.items()})
    if fmt == "label_dist":
        return json.dumps(LABEL_DIST)
    if fmt == "all":
        return json.dumps({
            "dutch_to_int":  DUTCH_TO_INT,
            "int_to_english": {str(k): v for k, v in INT_TO_ENGLISH.items()},
            "label_map":     {str(k): v for k, v in LABEL_MAP.items()},
            "label_dist":    {str(k): v for k, v in LABEL_DIST.items()},
            "num_classes":   NUM_CLASSES,
            "empty_value":   EMPTY_VALUE,
        }, indent=2)
    raise ValueError(f"Unknown --dump format '{fmt}'. Choose from: {_DUMP_FORMATS}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dump class-dictionary lookups for use in shell scripts."
    )
    parser.add_argument(
        "--dump",
        choices=_DUMP_FORMATS,
        default="all",
        help="Which lookup to print (default: all)",
    )
    args = parser.parse_args()
    print(_dump(args.dump))

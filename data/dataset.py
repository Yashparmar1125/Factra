from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


LABEL_MAP = {
    # canonical dataset labels
    "REAL": 0,
    "FAKE": 1,
    # common numeric exports (REAL->0, FAKE->1)
    "0": 0,
    "1": 1,
}


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Minimal, reproducible normalization:
    - remove URLs
    - remove special characters
    - lowercase
    """
    if not isinstance(text, str):
        return ""
    x = text.strip()
    x = _URL_RE.sub(" ", x)
    x = x.lower()
    x = _NON_ALNUM_RE.sub(" ", x)
    x = _MULTISPACE_RE.sub(" ", x).strip()
    return x


@dataclass(frozen=True)
class DatasetStats:
    n_rows: int
    n_null_text: int
    n_null_label: int
    class_counts: dict
    class_ratio: dict


import csv
import random

def load_csv_data(csv_paths: List[str], shuffle: bool = True, seed: int = 42) -> Tuple[List[str], List[int], DatasetStats]:
    """
    Pure Python CSV loader and dataset cleaner.
    Bypasses C-extension DLL blocks (Smart App Control) while providing
    the exact same data cleaning, normalization, and stats.
    """
    texts: List[str] = []
    labels: List[int] = []
    n_rows_raw = 0
    n_null_text = 0
    n_null_label = 0

    rows: List[Tuple[str, int]] = []

    for path in csv_paths:
        with open(path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                n_rows_raw += 1
                raw_text = r.get("text")
                raw_label = r.get("label")

                if raw_text is None or not str(raw_text).strip():
                    n_null_text += 1
                    continue
                if raw_label is None or not str(raw_label).strip():
                    n_null_label += 1
                    continue

                label_str = str(raw_label).strip().upper()
                if label_str not in LABEL_MAP:
                    n_null_label += 1
                    continue

                label_val = LABEL_MAP[label_str]
                cleaned = clean_text(raw_text)
                if not cleaned:
                    n_null_text += 1
                    continue

                rows.append((cleaned, label_val))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    c0 = sum(1 for _, l in rows if l == 0)
    c1 = sum(1 for _, l in rows if l == 1)
    total = len(rows)

    stats = DatasetStats(
        n_rows=total,
        n_null_text=n_null_text,
        n_null_label=n_null_label,
        class_counts={"0": c0, "1": c1},
        class_ratio={"0": c0 / total if total else 0.0, "1": c1 / total if total else 0.0},
    )

    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]
    return texts, labels, stats


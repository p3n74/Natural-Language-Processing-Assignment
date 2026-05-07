#!/usr/bin/env python3
"""Generate a tiny synthetic CSV that mimics Letterboxd columns for smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NEG = [
    "this film was dull and poorly paced i walked out bored",
    "terrible acting and a messy plot would not recommend",
    "waste of time confusing story and flat characters",
    "overlong tedious and forgettable completely disappointed",
    "bad script awkward dialogue nothing worked for me",
]

POS = [
    "beautiful cinematography and moving performances loved it",
    "tight script hilarious and heartfelt a joy to watch",
    "masterful direction every scene felt intentional stunning",
    "one of my favorites this year deeply affecting story",
    "sharp witty dialogue and great chemistry between leads",
]


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(160):
        template = str(rng.choice(NEG))
        noise = " ".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=3).tolist())
        rating = float(rng.choice([1.0, 1.5, 2.0]))
        rows.append({"Review": f"{template} {noise}", "Rating": rating})
    for _ in range(160):
        template = str(rng.choice(POS))
        noise = " ".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=3).tolist())
        rating = float(rng.choice([4.0, 4.5, 5.0]))
        rows.append({"Review": f"{template} {noise}", "Rating": rating})

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = Path("data/raw/sample_letterboxd_like.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

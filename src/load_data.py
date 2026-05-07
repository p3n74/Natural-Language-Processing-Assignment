"""Load Letterboxd-like CSV, map ratings to binary sentiment, split stratified."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_CANDIDATES = (
    "review",
    "Review",
    "review_text",
    "text",
    "Review Text",
    "reviewText",
    "body",
    "comment",
)
RATING_CANDIDATES = ("rating", "Rating", "stars", "Stars", "score", "star_rating")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    raise ValueError(
        f"No matching column found among {candidates}. Columns: {list(df.columns)}"
    )


def _parse_star_string(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    full = s.count("\u2605")  # ★
    half = s.count("\u00bd")  # ½
    if full == 0 and half == 0:
        return float("nan")
    return full + 0.5 * half


def _normalize_rating(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < max(1, int(0.5 * len(series))):
        numeric = series.map(_parse_star_string)
    mx = float(numeric.max(skipna=True)) if numeric.notna().any() else 5.0
    if mx > 5.5:
        numeric = numeric / 2.0
    return numeric.clip(0.5, 5.0)


def map_sentiment(ratings: pd.Series) -> pd.Series:
    r = _normalize_rating(ratings)
    out = pd.Series(index=r.index, dtype="float")
    out[r <= 2.0] = 0
    out[r >= 4.0] = 1
    return out


def balance_binary(df: pd.DataFrame, label_col: str, max_per_class: int | None, seed: int) -> pd.DataFrame:
    counts = {lab: int((df[label_col] == lab).sum()) for lab in (0, 1)}
    target = min(counts.values())
    if max_per_class is not None and max_per_class > 0:
        target = min(target, max_per_class)
    parts = []
    for lab in (0, 1):
        sub = df[df[label_col] == lab]
        if len(sub) > target:
            sub = sub.sample(n=target, random_state=seed)
        parts.append(sub)
    out = pd.concat(parts, axis=0)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits from Letterboxd CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-per-class", type=int, default=5000, help="Cap per class after balancing (None = no cap).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    n_raw = len(df)
    text_col = _pick_column(df, TEXT_CANDIDATES)
    rating_col = _pick_column(df, RATING_CANDIDATES)

    work = pd.DataFrame({"text": df[text_col], "rating": df[rating_col]})

    work["text"] = work["text"].astype(str).str.strip()
    n_after_blank = work[~work["text"].isin(["", "nan", "None"])]
    blank_dropped = len(work) - len(n_after_blank)
    work = n_after_blank

    work["text_norm"] = (
        work["text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )
    before_dedup = len(work)
    work = work.drop_duplicates(subset=["text_norm"]).drop(columns=["text_norm"])
    duplicates_dropped = before_dedup - len(work)

    work["label"] = map_sentiment(work["rating"])
    label_nan = int(work["label"].isna().sum())
    work = work.dropna(subset=["label"])

    cap = args.max_per_class if args.max_per_class and args.max_per_class > 0 else None
    pre_balance = {0: int((work["label"] == 0).sum()), 1: int((work["label"] == 1).sum())}
    work = balance_binary(work, "label", cap, args.seed)

    X = work["text"]
    y = work["label"].astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111111, random_state=args.seed, stratify=y_temp
    )

    def save_split(name: str, Xt: pd.Series, yt: pd.Series) -> None:
        out = pd.DataFrame({"text": Xt.values, "label": yt.values})
        path = args.output_dir / f"{name}.csv"
        out.to_csv(path, index=False)

    save_split("train", X_train, y_train)
    save_split("val", X_val, y_val)
    save_split("test", X_test, y_test)

    meta = {
        "source": str(args.input.resolve()),
        "n_raw_rows": n_raw,
        "blank_text_dropped": blank_dropped,
        "duplicates_dropped": duplicates_dropped,
        "label_unmapped_dropped": label_nan,
        "pre_balance_negatives": pre_balance[0],
        "pre_balance_positives": pre_balance[1],
        "n_after_balance": int(len(work)),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    (args.output_dir / "split_meta.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()), encoding="utf-8")
    print(f"Wrote splits to {args.output_dir} ({meta})")


if __name__ == "__main__":
    main()

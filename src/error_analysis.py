"""Sample misclassified reviews for qualitative error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ARTIFACTS = Path("artifacts")
OUT_DIR = ARTIFACTS


def main() -> None:
    test_path = Path("data/processed/test.csv")
    best_path = ARTIFACTS / "best_model.json"
    manifest_path = ARTIFACTS / "models_manifest.json"
    if not test_path.exists():
        raise SystemExit(f"Missing {test_path}")
    if not best_path.exists():
        raise SystemExit("Missing best_model.json; run python -m src.evaluate first.")

    best_name = json.loads(best_path.read_text(encoding="utf-8"))["best_model"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((m for m in manifest if m["name"] == best_name), None)
    if entry is None:
        raise SystemExit(f"Model {best_name} not in manifest")

    pipe = joblib.load(ARTIFACTS / entry["file"])
    df = pd.read_csv(test_path)
    X = df["text"].astype(str).values
    y = df["label"].astype(int).values
    pred = pipe.predict(X)

    view = df.copy()
    view["pred"] = pred
    wrong = view[view["label"] != view["pred"]].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if wrong.empty:
        placeholder = pd.DataFrame(columns=["text", "label", "pred", "snippet"])
        placeholder.to_csv(OUT_DIR / "error_analysis_samples.csv", index=False)
        (OUT_DIR / "error_analysis_table.md").write_text(
            "# Error analysis\n\nNo misclassified examples on the test split (often happens on toy data).\n",
            encoding="utf-8",
        )
        print("No misclassifications; wrote empty placeholder files.")
        return

    wrong["snippet"] = wrong["text"].str.slice(0, 280)

    sample = wrong.head(40)
    sample.to_csv(OUT_DIR / "error_analysis_samples.csv", index=False)

    lines = [
        "# Error analysis sample (best model: %s)\n" % best_name,
        "| true | pred | preview |",
        "| --- | --- | --- |",
    ]
    for _, row in sample.head(15).iterrows():
        prev = row["snippet"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['label']} | {row['pred']} | {prev} |")
    (OUT_DIR / "error_analysis_table.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_DIR / 'error_analysis_samples.csv'} ({len(sample)} rows preview)")


if __name__ == "__main__":
    main()

"""Evaluate saved models on the held-out test set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ARTIFACTS = Path("artifacts")
FIG_DIR = Path("report/figures")


def _scores_binary(y_true, y_pred, y_score) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_neg": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_neg": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_neg": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = None
    return out


def _decision_or_proba(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def main() -> None:
    test_path = Path("data/processed/test.csv")
    manifest_path = ARTIFACTS / "models_manifest.json"
    if not test_path.exists():
        raise SystemExit(f"Missing {test_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}; run python -m src.train first.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_path)
    X_test = df["text"].astype(str).values
    y_test = df["label"].astype(int).values

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []

    for entry in manifest:
        name = entry["name"]
        path = ARTIFACTS / entry["file"]
        pipe = joblib.load(path)
        y_pred = pipe.predict(X_test)
        y_score = _decision_or_proba(pipe, X_test)
        metrics = _scores_binary(y_test, y_pred, y_score)
        metrics["model"] = name
        rows.append(metrics)

        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        disp.ax_.set_title(name)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confusion_{name}.png", dpi=150)
        plt.close()

        if metrics.get("roc_auc") is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title(f"ROC — {name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"roc_{name}.png", dpi=150)
            plt.close()

    summary = pd.DataFrame(rows)
    summary_path = ARTIFACTS / "metrics_test.csv"
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(8, 4))
    melted = summary.melt(id_vars=["model"], value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    sns.barplot(data=melted, x="model", y="value", hue="variable")
    plt.xticks(rotation=20)
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "metrics_bar.png", dpi=150)
    plt.close()

    report_txt = []
    for entry in manifest:
        name = entry["name"]
        pipe = joblib.load(ARTIFACTS / entry["file"])
        y_pred = pipe.predict(X_test)
        report_txt.append(f"=== {name} ===\n")
        report_txt.append(classification_report(y_test, y_pred, digits=3))
        report_txt.append("\n")
    (ARTIFACTS / "classification_reports.txt").write_text("".join(report_txt), encoding="utf-8")

    best = summary.sort_values("f1_macro", ascending=False).iloc[0]
    pick = {"best_model": best["model"], "f1_macro": float(best["f1_macro"])}
    (ARTIFACTS / "best_model.json").write_text(json.dumps(pick, indent=2), encoding="utf-8")

    print(summary)
    print(f"Wrote {summary_path}, figures under {FIG_DIR}, best model {pick}")


if __name__ == "__main__":
    main()

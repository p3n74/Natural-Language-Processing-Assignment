"""Train baseline and TF-IDF models with small hyperparameter search."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.features import bow_pipeline, tfidf_pipeline

ARTIFACTS = Path("artifacts")


def _cv_splits(y: np.ndarray, max_splits: int = 5) -> int:
    counts = np.bincount(y.astype(int))
    return max(2, min(max_splits, int(counts.min()), len(y) // max_splits))


def train_one(name: str, pipe, param_grid: dict, X_train, y_train, seed: int) -> GridSearchCV:
    cv_n = _cv_splits(y_train)
    cv = StratifiedKFold(n_splits=cv_n, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, ARTIFACTS / f"model_{name}.joblib")
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
    row = {"model": name, "best_params": best, "best_cv_f1_macro": float(gs.best_score_)}
    print(row)
    return gs


def main() -> None:
    seed = 42
    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        raise SystemExit(f"Missing {train_path}; run python -m src.load_data first.")

    df = pd.read_csv(train_path)
    X_train = df["text"].astype(str).values
    y_train = df["label"].astype(int).values

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifest = []

    lr_pipe = tfidf_pipeline(LogisticRegression(max_iter=4000, solver="liblinear", random_state=seed))
    lr_grid = {
        "vec__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }
    gs_lr = train_one("lr_tfidf", lr_pipe, lr_grid, X_train, y_train, seed)
    manifest.append({"name": "lr_tfidf", "file": "model_lr_tfidf.joblib", "cv_f1_macro": float(gs_lr.best_score_)})

    nb_pipe = tfidf_pipeline(MultinomialNB())
    nb_grid = {"vec__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 1.0, 2.0]}
    gs_nb = train_one("nb_tfidf", nb_pipe, nb_grid, X_train, y_train, seed)
    manifest.append({"name": "nb_tfidf", "file": "model_nb_tfidf.joblib", "cv_f1_macro": float(gs_nb.best_score_)})

    svm_pipe = tfidf_pipeline(LinearSVC(random_state=seed))
    svm_grid = {"vec__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 1.0, 10.0]}
    gs_svm = train_one("svm_tfidf", svm_pipe, svm_grid, X_train, y_train, seed)
    manifest.append({"name": "svm_tfidf", "file": "model_svm_tfidf.joblib", "cv_f1_macro": float(gs_svm.best_score_)})

    nb_bow = bow_pipeline(MultinomialNB())
    nb_bow_grid = {"vec__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 1.0, 2.0]}
    gs_nb_bow = train_one("nb_bow", nb_bow, nb_bow_grid, X_train, y_train, seed)
    manifest.append({"name": "nb_bow", "file": "model_nb_bow.joblib", "cv_f1_macro": float(gs_nb_bow.best_score_)})

    (ARTIFACTS / "models_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest with {len(manifest)} models under {ARTIFACTS}")


if __name__ == "__main__":
    main()

# Letterboxd review sentiment (NLP course project)

Binary sentiment classification on movie-review text to satisfy the rubric in `guide.md`. Pipeline: load → preprocess → TF-IDF / BoW → Logistic Regression, Multinomial Naive Bayes, Linear SVM → metrics, confusion matrices, error analysis. Final write-up: `report/main.tex` → PDF.

## Quick start (sample data)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional notebook stack:
pip install -r requirements-dev.txt
python -m src.setup_nltk
make sample && make data && make train && make evaluate && make errors
```

With a real Kaggle CSV, place it under `data/raw/` and run:

```bash
python -m src.load_data --input data/raw/your_export.csv --max-per-class 5000
python -m src.train
python -m src.evaluate
python -m src.error_analysis
```

Recommended dataset: [Letterboxd Movie Reviews (~90,000)](https://www.kaggle.com/datasets/riyosha/letterboxd-movie-reviews-90000) (Riyosha on Kaggle). Map columns automatically if they contain review text and numeric star ratings.

## PDF report

Requires a LaTeX distribution (TeX Live) with `latexmk`:

```bash
make report
```

## Optional scrape

Small supplemental fetch from public Letterboxd HTML (rate-limited):

```bash
python -m src.scrape_letterboxd --urls "https://letterboxd.com/film/example/reviews/" --max 30 --output data/raw/scraped_reviews.csv
```

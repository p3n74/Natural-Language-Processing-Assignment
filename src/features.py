"""Feature pipelines: TF-IDF and Bag-of-Words after NLTK preprocessing."""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.preprocess import preprocess_text


def _prep_batch(X):
    return [preprocess_text(str(x)) for x in X]


def make_prep_transformer() -> FunctionTransformer:
    return FunctionTransformer(_prep_batch, validate=False)


def tfidf_pipeline(classifier, **tfidf_kw) -> Pipeline:
    defaults = dict(min_df=3, max_features=50000, sublinear_tf=True)
    defaults.update(tfidf_kw)
    vec = TfidfVectorizer(**defaults)
    return Pipeline([("prep", make_prep_transformer()), ("vec", vec), ("clf", classifier)])


def bow_pipeline(classifier, **bow_kw) -> Pipeline:
    defaults = dict(min_df=3, max_features=50000)
    defaults.update(bow_kw)
    vec = CountVectorizer(**defaults)
    return Pipeline([("prep", make_prep_transformer()), ("vec", vec), ("clf", classifier)])

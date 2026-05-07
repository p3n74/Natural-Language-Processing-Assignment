"""Text preprocessing: tokenize, lowercase, stopwords, lemmatize, strip noise."""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_LEMMATIZER = WordNetLemmatizer()
_STOP: set[str] | None = None
_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY, _STOP
    if _NLTK_READY:
        return
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            continue
    _STOP = set(stopwords.words("english"))
    _NLTK_READY = True


def preprocess_text(text: str) -> str:
    _ensure_nltk()
    assert _STOP is not None
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in _STOP]
    lemmas = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

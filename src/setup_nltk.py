"""NLTK data download for tokenization, stopwords, and lemmatization."""

import nltk


def main() -> None:
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=False)
        except Exception as exc:
            print(f"warning: could not download {pkg}: {exc}")


if __name__ == "__main__":
    main()

"""Optional small scrape of public Letterboxd review snippets (polite, rate-limited)."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "AcademicNLPProject/1.0 (+research)"}


def parse_rating_li(li) -> float | None:
    rating_span = li.select_one("span.rating")
    if rating_span is None:
        return None
    cls = rating_span.get("class") or []
    for c in cls:
        m = re.match(r"rated-(\d+)-stars?", str(c))
        if m:
            return int(m.group(1)) / 2.0
    return None


def parse_review_body(li) -> str:
    body = li.select_one(".review-body")
    if body:
        return body.get_text(" ", strip=True)
    return li.get_text(" ", strip=True)


def fetch_page(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def scrape_list(url: str, max_reviews: int, pause: float) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    page_url = url
    while page_url and len(rows) < max_reviews:
        soup = fetch_page(page_url)
        items = soup.select("li.film-detail")
        for li in items:
            if len(rows) >= max_reviews:
                break
            rid = li.get("data-item-id") or str(hash(li.get_text()))
            if rid in seen:
                continue
            seen.add(rid)
            rating = parse_rating_li(li)
            text = parse_review_body(li)
            if rating is None or not text:
                continue
            rows.append({"rating": rating, "review": text})
        next_a = soup.select_one("a.next")
        page_url = ""
        if next_a and next_a.get("href"):
            host = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(url))
            page_url = host + next_a["href"]
        time.sleep(pause)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape a small number of Letterboxd reviews.")
    parser.add_argument("--urls", nargs="+", required=True, help="Film review listing URLs.")
    parser.add_argument("--max", type=int, default=40, dest="max_reviews")
    parser.add_argument("--output", type=Path, default=Path("data/raw/scraped_reviews.csv"))
    parser.add_argument("--pause", type=float, default=1.5, help="Seconds between HTTP requests.")
    args = parser.parse_args()

    all_rows: list[dict] = []
    per = max(1, args.max_reviews // max(1, len(args.urls)))
    for u in args.urls:
        all_rows.extend(scrape_list(u, per, args.pause))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rating", "review"])
        w.writeheader()
        w.writerows(all_rows[: args.max_reviews])
    print(f"Wrote {len(all_rows[: args.max_reviews])} rows to {args.output}")


if __name__ == "__main__":
    main()

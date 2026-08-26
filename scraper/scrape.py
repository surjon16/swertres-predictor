"""
Scrapes historical Swertres (3D Lotto) results from pinoyswertres.net's
year-archive pages and writes them to data/swertres_history.csv.

Source page structure (verified 2026-08-26):
  https://www.pinoyswertres.net/swertres-result-history-{YEAR}/
  <table class="has-fixed-layout">
    <thead><tr><th>Draw Date</th><th>2:00 PM</th><th>5:00 PM</th><th>9:00 PM</th></tr></thead>
    <tbody><tr><td><a>Jan 1, 2025</a></td><td>5-7-8</td><td>3-4-2</td><td>6-4-5</td></tr> ...

Missing draws are marked with "*" on the source site and are skipped.
"""
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE_URL = "https://www.pinoyswertres.net/swertres-result-history-{year}/"
CURRENT_YEAR_URL = "https://www.pinoyswertres.net/swertres-result-history/"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (research scraper; contact via repo issues)"}

ROW_RE = re.compile(
    r'<tr><td[^>]*><a[^>]*>([^<]+)</a></td>'
    r'<td[^>]*>([^<]*)</td>'
    r'<td[^>]*>([^<]*)</td>'
    r'<td[^>]*>([^<]*)</td></tr>',
    re.IGNORECASE,
)

DRAW_COLUMNS = ["2PM", "5PM", "9PM"]


def fetch(url: str) -> str:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_year_page(html: str, year: int) -> list[dict]:
    rows = []
    for date_str, pm2, pm5, pm9 in ROW_RE.findall(html):
        try:
            date = datetime.strptime(date_str.strip(), "%b %d, %Y").date()
        except ValueError:
            continue
        if date.year != year:
            continue
        for col, raw in zip(DRAW_COLUMNS, (pm2, pm5, pm9)):
            raw = raw.strip()
            if not raw or "*" in raw or "-" not in raw:
                continue
            digits = raw.split("-")
            if len(digits) != 3 or not all(d.isdigit() for d in digits):
                continue
            rows.append(
                {
                    "date": date.isoformat(),
                    "draw_time": col,
                    "d1": digits[0],
                    "d2": digits[1],
                    "d3": digits[2],
                    "combo": "".join(digits),
                }
            )
    return rows


def scrape_years(start_year: int, end_year: int) -> list[dict]:
    all_rows = []
    for year in range(start_year, end_year + 1):
        url = BASE_URL.format(year=year)
        print(f"Fetching {year} -> {url}", file=sys.stderr)
        try:
            html = fetch(url)
        except (HTTPError, URLError) as e:
            print(f"  skip {year}: {e}", file=sys.stderr)
            continue
        rows = parse_year_page(html, year)
        print(f"  parsed {len(rows)} draws", file=sys.stderr)
        all_rows.extend(rows)
        time.sleep(1)  # be polite
    return all_rows


def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2009
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else datetime.now().year

    rows = scrape_years(start_year, end_year)
    rows.sort(key=lambda r: (r["date"], DRAW_COLUMNS.index(r["draw_time"])))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "draw_time", "d1", "d2", "d3", "combo"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()

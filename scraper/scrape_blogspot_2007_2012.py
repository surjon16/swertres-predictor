"""
Scrapes iloveswertres.blogspot.com's year pages (2007-2012), which use a
calendar-grid table format pasted from Excel/Word:

  Row 0: DATE | Jan {year} (colspan=N) | Feb {year} (colspan=N) | ...
  Row 1: (blank) | <slot labels for Jan, N of them> | <slot labels for Feb> | ...
  Rows 2-32: day-of-month (1-31), then one result cell per (month, slot).

The number of daily draw slots and their labels VARY BY YEAR -- this is a
real historical fact, not a bug: 2007 had 2 draws/day (2PM, 9PM), 2008-2010
had 3 (2PM, 5PM, 9PM), 2011-2012 had 3 but at different times (11AM, 4PM,
9PM). Draw-time labels are preserved as-recorded rather than normalized,
since the underlying slot isn't necessarily the same across eras.

Only 2007-2012 pages exist on this blog (verified 2026-08-26); other year
URLs 404.
"""
import csv
import sys
import time
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from bs4 import BeautifulSoup

BASE_URL = "https://iloveswertres.blogspot.com/p/{year}-results.html"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history_2007_2012.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (research scraper; contact via repo issues)"}

MONTH_NAMES = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]


def fetch(url: str) -> str:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_year_page(html: str, year: int) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")

    header0 = rows[0].find_all(["td", "th"])[1:]  # skip DATE cell
    header1 = [c.get_text(strip=True).upper() for c in rows[1].find_all(["td", "th"])]

    # Build a list of (month_index, slot_label) for every data column, in order.
    col_meta = []
    col_cursor = 0
    for month_cell in header0:
        colspan = int(month_cell.get("colspan", 1))
        month_text = month_cell.get_text(strip=True).lower()[:3]
        month_idx = MONTH_NAMES.index(month_text) + 1
        for _ in range(colspan):
            col_meta.append((month_idx, header1[col_cursor]))
            col_cursor += 1

    out_rows = []
    for tr in rows[2:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        day_text = cells[0].get_text(strip=True)
        if not day_text.isdigit():
            continue
        day = int(day_text)
        data_cells = cells[1:]
        for (month_idx, slot_label), cell in zip(col_meta, data_cells):
            raw = cell.get_text(strip=True)
            if not raw.isdigit() or len(raw) != 3:
                continue
            try:
                d = date(year, month_idx, day)
            except ValueError:
                continue  # e.g. Feb 30 doesn't exist, skip
            out_rows.append(
                {
                    "date": d.isoformat(),
                    "draw_time": slot_label,
                    "d1": raw[0],
                    "d2": raw[1],
                    "d3": raw[2],
                    "combo": raw,
                }
            )
    return out_rows


def main():
    years = range(2007, 2013)  # 2007-2012 inclusive; other years 404 on this source
    all_rows = []
    for year in years:
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
        time.sleep(1)

    all_rows.sort(key=lambda r: r["date"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "draw_time", "d1", "d2", "d3", "combo"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()

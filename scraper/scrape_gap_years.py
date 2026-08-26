"""
Fills the 2002-2004, 2005-2006, and 2013-2021 gap using user-supplied
URLs. Three distinct page formats are involved:

FORMAT A -- single-year calendar grid (iloveswertres.blogspot.com,
  2013/2014/2015/2016/2017/2018("blog-page_28")/2019/2021):
  Same structure as scrape_blogspot_2007_2012.py: row0 = month headers
  (colspan=N slots each), row1 = flat slot labels, data from row2.
  Year is parsed out of the month-header text (e.g. "JAN 2013") rather
  than trusted from the URL, since slugs are irregular.

FORMAT B -- multi-year side-by-side blocks (iloveswertres.blogspot.com,
  2002-2004 and 2005-2006 combined pages):
  Each page packs 2-3 years side by side as separate blocks, separated by
  a blank spacer column. Each block has its own "date" header cell with
  rowspan=3 (year block spans 3 header rows: year label, month names,
  slot labels) instead of rowspan=2. Individual month cells in the month
  row can themselves have colspan>1 when a new draw slot was introduced
  mid-year (e.g. Nov/Dec 2006 split into 2PM+9PM after Nov 8, 2006).

FORMAT C -- pinoyswertres.net's older single-table-with-month-header-rows
  style (2020 page): one flat table where a row containing a month name
  (e.g. "JANUARY") acts as a sub-header defining slot columns for the
  following day rows, until the next month-header row. Dates are
  M/D/YYYY, combos are hyphenated "D-D-D", missing draws marked "N/D".
"""
import csv
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from bs4 import BeautifulSoup

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history_gap_years.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (research scraper; contact via repo issues)"}

MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_FULL = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

FORMAT_A_URLS = [
    "https://iloveswertres.blogspot.com/p/2013.html",
    "https://iloveswertres.blogspot.com/p/2014.html",
    "https://iloveswertres.blogspot.com/p/2015.html",
    "https://iloveswertres.blogspot.com/p/2016_56.html",
    "https://iloveswertres.blogspot.com/p/2017.html",
    "https://iloveswertres.blogspot.com/p/blog-page_28.html",  # 2018
    "https://iloveswertres.blogspot.com/p/2019.html",
    "https://iloveswertres.blogspot.com/p/2021.html",
]
FORMAT_B_URLS = [
    "https://iloveswertres.blogspot.com/p/200420032002-results.html",
    "https://iloveswertres.blogspot.com/p/20062005-results.html",
]
FORMAT_C_URLS = [
    "https://www.pinoyswertres.net/pcso-swertres-result-history-2020/",
]


def fetch(url: str) -> str:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_format_a(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")

    header0 = rows[0].find_all(["td", "th"])[1:]
    header1 = [c.get_text(strip=True).upper() for c in rows[1].find_all(["td", "th"])]

    col_meta = []
    col_cursor = 0
    for month_cell in header0:
        text = month_cell.get_text(strip=True)
        colspan = int(month_cell.get("colspan", 1))
        month_abbr = text.lower()[:3]
        if month_abbr not in MONTH_NAMES:
            continue
        month_idx = MONTH_NAMES.index(month_abbr) + 1
        year_match = re.search(r"(19|20)\d{2}", text)
        year = int(year_match.group()) if year_match else None
        for _ in range(colspan):
            if col_cursor < len(header1):
                col_meta.append((year, month_idx, header1[col_cursor]))
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
        for (year, month_idx, slot_label), cell in zip(col_meta, cells[1:]):
            if year is None:
                continue
            raw = cell.get_text(strip=True)
            if not raw.isdigit() or len(raw) != 3:
                continue
            try:
                d = date(year, month_idx, day)
            except ValueError:
                continue
            out_rows.append({
                "date": d.isoformat(), "draw_time": slot_label,
                "d1": raw[0], "d2": raw[1], "d3": raw[2], "combo": raw,
            })
    return out_rows


def parse_format_b(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")

    row0 = rows[0].find_all(["td", "th"])
    blocks = []  # list of (year, width)
    i = 0
    while i < len(row0):
        cell = row0[i]
        text = cell.get_text(strip=True).lower()
        if text == "date":
            year_cell = row0[i + 1]
            year_match = re.search(r"(19|20)\d{2}", year_cell.get_text(strip=True))
            year = int(year_match.group()) if year_match else None
            width = int(year_cell.get("colspan", 1))
            blocks.append([year, width])
            i += 2
        else:
            i += 1  # spacer or anything unexpected

    def consume_row_into_blocks(cells):
        """Split a flat cell list into per-block chunks of `width`, skipping
        one spacer cell between consecutive blocks."""
        chunks = []
        idx = 0
        for b_i, (year, width) in enumerate(blocks):
            chunk = []
            consumed = 0
            while consumed < width and idx < len(cells):
                cell = cells[idx]
                span = int(cell.get("colspan", 1))
                for _ in range(span):
                    chunk.append(cell)
                    consumed += 1
                idx += 1
            chunks.append(chunk)
            if b_i < len(blocks) - 1 and idx < len(cells):
                idx += 1  # skip spacer cell
        return chunks

    month_row_cells = rows[1].find_all(["td", "th"])
    month_chunks = consume_row_into_blocks(month_row_cells)

    slot_row_cells = rows[2].find_all(["td", "th"])
    slot_chunks = consume_row_into_blocks(slot_row_cells)

    # col_meta[b] = list of (month_idx, slot_label) per physical column in block b
    col_meta = []
    for b_i, (year, width) in enumerate(blocks):
        months = month_chunks[b_i]
        slots = slot_chunks[b_i]
        meta = []
        for j in range(width):
            month_text = months[j].get_text(strip=True).lower()[:3] if j < len(months) else ""
            slot_text = slots[j].get_text(strip=True).upper() if j < len(slots) else ""
            month_idx = MONTH_NAMES.index(month_text) + 1 if month_text in MONTH_NAMES else None
            meta.append((month_idx, slot_text))
        col_meta.append(meta)

    out_rows = []
    for tr in rows[3:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        idx = 0
        for b_i, (year, width) in enumerate(blocks):
            if idx >= len(cells):
                break
            day_text = cells[idx].get_text(strip=True)
            idx += 1
            body = cells[idx: idx + width]
            idx += width
            if b_i < len(blocks) - 1:
                idx += 1  # spacer
            if year is None or not day_text.isdigit():
                continue
            day = int(day_text)
            for (month_idx, slot_label), cell in zip(col_meta[b_i], body):
                if month_idx is None:
                    continue
                raw = cell.get_text(strip=True)
                if not raw.isdigit() or len(raw) != 3:
                    continue
                try:
                    d = date(year, month_idx, day)
                except ValueError:
                    continue
                out_rows.append({
                    "date": d.isoformat(), "draw_time": slot_label,
                    "d1": raw[0], "d2": raw[1], "d3": raw[2], "combo": raw,
                })
    return out_rows


def parse_format_c(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")

    out_rows = []
    current_slots = None
    for tr in rows:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        first_text = cells[0].get_text(strip=True).lower()
        if first_text in MONTH_FULL:
            current_slots = [c.get_text(strip=True).upper() for c in cells[1:]]
            # Source has a typo: September 2020's header says "2AM" -- no such
            # draw slot exists; every other month in this schedule era reads
            # "2PM" (matching the pre/post transition slots), so normalize it.
            current_slots = ["2PM" if s == "2AM" else s for s in current_slots]
            continue
        date_text = cells[0].get_text(strip=True)
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_text)
        if not m or current_slots is None:
            continue
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        for slot_label, cell in zip(current_slots, cells[1:]):
            raw = cell.get_text(strip=True)
            digit_match = re.match(r"^(\d)-(\d)-(\d)$", raw)
            if not digit_match:
                continue
            d1, d2, d3 = digit_match.groups()
            out_rows.append({
                "date": d.isoformat(), "draw_time": slot_label,
                "d1": d1, "d2": d2, "d3": d3, "combo": f"{d1}{d2}{d3}",
            })
    return out_rows


def main():
    all_rows = []

    for url in FORMAT_A_URLS:
        print(f"[A] Fetching {url}", file=sys.stderr)
        try:
            html = fetch(url)
        except (HTTPError, URLError) as e:
            print(f"  skip: {e}", file=sys.stderr)
            continue
        rows = parse_format_a(html)
        print(f"  parsed {len(rows)} draws", file=sys.stderr)
        all_rows.extend(rows)
        time.sleep(1)

    for url in FORMAT_B_URLS:
        print(f"[B] Fetching {url}", file=sys.stderr)
        try:
            html = fetch(url)
        except (HTTPError, URLError) as e:
            print(f"  skip: {e}", file=sys.stderr)
            continue
        rows = parse_format_b(html)
        print(f"  parsed {len(rows)} draws", file=sys.stderr)
        all_rows.extend(rows)
        time.sleep(1)

    for url in FORMAT_C_URLS:
        print(f"[C] Fetching {url}", file=sys.stderr)
        try:
            html = fetch(url)
        except (HTTPError, URLError) as e:
            print(f"  skip: {e}", file=sys.stderr)
            continue
        rows = parse_format_c(html)
        print(f"  parsed {len(rows)} draws", file=sys.stderr)
        all_rows.extend(rows)
        time.sleep(1)

    all_rows.sort(key=lambda r: (r["date"], r["draw_time"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "draw_time", "d1", "d2", "d3", "combo"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()

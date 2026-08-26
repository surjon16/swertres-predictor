# swertres-predictor

Historical Swertres (PH 3D Lotto) data collection and statistical bias
analysis. Goal: test whether physical draw-equipment wear/bias produces
detectable digit-frequency skew, using chi-square goodness-of-fit tests
per draw slot (2PM / 5PM / 9PM) and digit position (1st/2nd/3rd).

This is NOT a "predict the winning number" tool — a fair lottery draw is
independent per draw and cannot be predicted. It's a statistical test for
whether the *equipment* has a measurable, non-uniform bias.

## Data

Three sources, three scripts (plus a merge step):

- `scraper/scrape.py` — pinoyswertres.net year-archive pages.
  Despite the site listing archive links back to 2009, only 2022 onward
  actually resolves (2009-2021 links 301-redirect to the homepage — no
  data behind them). Real coverage: 2022-01-01 through today, ~3,300
  draws, slots 2PM/5PM/9PM. `data/swertres_history.csv`.

      python3 scraper/scrape.py 2022 2026

- `scraper/scrape_blogspot_2007_2012.py` — iloveswertres.blogspot.com
  year pages. Only 2007-2012 exist on this source (other years 404).
  ~6,000 draws. `data/swertres_history_2007_2012.csv`.

      python3 scraper/scrape_blogspot_2007_2012.py

- `scraper/scrape_gap_years.py` — fills 2002-2006 and 2013-2021 from
  user-supplied URLs spanning 3 different page layouts on 2 sites
  (single-year calendar grids, multi-year side-by-side blocks, and
  pinoyswertres.net's older month-header-row style for 2020).
  ~10,900 draws. `data/swertres_history_gap_years.csv`.

      python3 scraper/scrape_gap_years.py

**The draw schedule itself changed over time** (this is real PCSO
history, not a data-quality issue):
  - 2002 (from Jun 13, Swertres launch) - 2006 (to Nov 7): 1 draw/day — 9PM only
  - 2006 (from Nov 8): 2nd draw added — 2PM, 9PM
  - 2008 (from ~mid-year) - 2010: 3 draws/day — 2PM, 5PM, 9PM
  - 2011-2020 (to ~Aug): 3 draws/day — 11AM, 4PM, 9PM
  - 2020 (from ~Aug) -present: 3 draws/day — 2PM, 5PM, 9PM

`draw_time` labels are preserved as-recorded per era rather than
normalized, since a slot at a different time of day may not be the same
underlying equipment/process. One source typo was corrected: a page
labeled one month's 2PM column "2AM" (no such draw exists; every
adjacent month uses "2PM" in the same schedule era).

**Real gap in the data, not a scraping artifact:** 2020-03-19 through
2020-08-23 has no draws recorded on any source — this lines up with the
Philippines' COVID-19 lockdown period, during which PCSO actually
suspended draws.

**Coverage is now continuous, 2002-2026, no missing years.** Combined:
`data/swertres_history_combined.csv`, ~20,200 draws total (merges all
three sources, deduplicated on date+draw_time — regenerate by re-running
all three scrapers and concatenating). Columns throughout:
`date, draw_time, d1, d2, d3, combo`.

## Analysis

TODO: chi-square uniformity test per digit position per draw slot.

Run:

    .venv/bin/python -m venv .venv  # if not already created
    .venv/bin/pip install scipy pandas
    .venv/bin/python analysis/bias_test.py

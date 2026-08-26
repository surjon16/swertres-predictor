# swertres-predictor

Historical Swertres (PH 3D Lotto) data collection and statistical bias
analysis. Goal: test whether physical draw-equipment wear/bias produces
detectable digit-frequency skew, using chi-square goodness-of-fit tests
per draw slot (2PM / 5PM / 9PM) and digit position (1st/2nd/3rd).

This is NOT a "predict the winning number" tool — a fair lottery draw is
independent per draw and cannot be predicted. It's a statistical test for
whether the *equipment* has a measurable, non-uniform bias.

## Data

Two sources, two scripts:

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

**The draw schedule itself changed over time** (this is real PCSO
history, not a data-quality issue):
  - 2007: 2 draws/day — 2PM, 9PM
  - 2008 (from ~mid-year) - 2010: 3 draws/day — 2PM, 5PM, 9PM
  - 2011-2012: 3 draws/day — 11AM, 4PM, 9PM
  - 2022-present: 3 draws/day — 2PM, 5PM, 9PM

`draw_time` labels are preserved as-recorded per era rather than
normalized, since a slot at a different time of day may not be the same
underlying equipment/process.

**Known gap:** 2013-2021 has no data from either source — neither site
has an archive covering those years. Combined coverage is 2007-2012 +
2022-present, ~9,300 draws total, with a 9-year hole in the middle.

`data/swertres_history_combined.csv` merges both sources (built via a
one-off pandas concat — regenerate by re-running both scrapers and
concatenating). Columns throughout: `date, draw_time, d1, d2, d3, combo`.

## Analysis

TODO: chi-square uniformity test per digit position per draw slot.

Run:

    .venv/bin/python -m venv .venv  # if not already created
    .venv/bin/pip install scipy pandas
    .venv/bin/python analysis/bias_test.py

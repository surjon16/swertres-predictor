# swertres-predictor

Historical Swertres (PH 3D Lotto) data collection and statistical bias
analysis. Goal: test whether physical draw-equipment wear/bias produces
detectable digit-frequency skew, using chi-square goodness-of-fit tests
per draw slot (2PM / 5PM / 9PM) and digit position (1st/2nd/3rd).

This is NOT a "predict the winning number" tool — a fair lottery draw is
independent per draw and cannot be predicted. It's a statistical test for
whether the *equipment* has a measurable, non-uniform bias.

## Data

`scraper/scrape.py` pulls historical results from pinoyswertres.net's
year-archive pages.

**Known limitation:** despite the source site listing archive links back
to 2009, only 2022 onward actually resolves (2009-2021 links 301-redirect
to the homepage — no data behind them). Real coverage is 2022-08-26
(2022-01-01 through today), ~3,300 draws.

    python3 scraper/scrape.py 2022 2026

Output: `data/swertres_history.csv` with columns
`date, draw_time, d1, d2, d3, combo`.

## Analysis

TODO: chi-square uniformity test per digit position per draw slot.

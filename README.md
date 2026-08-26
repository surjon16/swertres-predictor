# Swertres Predictor: A Statistical Investigation

**Research question:** Does the Philippine Swertres (3D Lotto) draw show any
statistically detectable, exploitable non-randomness — whether from
mechanical equipment bias, seasonal effects, or any other structure in the
historical record — that could inform a prediction strategy?

**Short answer: no.** Every method tested below, across ~20,200 historical
draws spanning 2002-2026, performed at or below the level expected from
pure chance. This repository documents the full investigation: the
hypothesis, the data-collection effort, every method tried, and the
reasoning behind each conclusion — including several points where a naive
read of the results would have been misleading.

This is **not** a "predict the winning number" tool. It's a record of a
rigorous attempt to find exploitable structure in a real-world gambling
process, and an honest account of coming up empty.

---

## 1. Background and motivation

Swertres is drawn using physical/mechanical equipment, operated by people,
repeatedly, over decades. The starting hypothesis was reasonable on its
face: **mechanical randomization devices can wear unevenly, and real-world
lotteries have documented cases of measurable equipment bias** (e.g. balls
or machines producing skewed number frequencies). Girolamo Cardano — the
16th-century mathematician who wrote the first systematic probability
treatise, *Liber de Ludo Aleae* — explicitly considered warped or weighted
dice as a real, detectable source of unfairness in games of chance. See
[`docs/cardano_probability_notes.md`](docs/cardano_probability_notes.md)
for the full historical grounding: Cardano's own toolkit (equiprobable
"circuits," fairness-via-frequency-deviation, the multiplication rule for
independent events, and an early law of large numbers) turns out to be
essentially the same toolkit used throughout this project, four centuries
early.

So the question was worth asking with real data and real statistics,
rather than dismissed outright. This repo is that investigation.

## 2. Data collection

Three sources were scraped, covering three different eras and three
different page formats, to build as complete a historical record as
possible:

| Script | Source | Coverage | Draws |
|---|---|---|---|
| `scraper/scrape.py` | pinoyswertres.net year-archive pages | 2022-present (site's own links to 2009-2021 are dead/redirect) | ~3,300 |
| `scraper/scrape_blogspot_2007_2012.py` | iloveswertres.blogspot.com | 2007-2012 | ~6,000 |
| `scraper/scrape_gap_years.py` | iloveswertres.blogspot.com + pinoyswertres.net (3 distinct page layouts) | 2002-2006, 2013-2021 | ~10,900 |

Combined and deduplicated: **`data/swertres_history_combined.csv`, ~20,200
draws, 2002-06-13 through today, zero missing years.**

### Real historical facts surfaced during collection (not data artifacts)

- **The draw schedule itself changed over time.** 2002-2006: 1 draw/day
  (9PM only). Nov 8, 2006: 2PM draw added. Mid-2008: 5PM draw added
  (2PM/5PM/9PM). 2011-~Aug 2020: schedule shifted to 11AM/4PM/9PM.
  ~Aug 2020-present: back to 2PM/5PM/9PM. `draw_time` labels are preserved
  as-recorded per era rather than normalized, since a slot at a different
  time of day is not necessarily the same underlying equipment.
- **A real ~5-month gap exists**: 2020-03-19 through 2020-08-23 has no
  draws on any source — this lines up exactly with the Philippines'
  COVID-19 lockdown, during which PCSO actually suspended draws.
- One source typo was found and corrected: a page mislabeled a column
  "2AM" where every adjacent month used "2PM" in the same schedule era.
- A merge bug was found and fixed: an early version of the combined-CSV
  build script read the `combo` column without forcing string type,
  silently dropping leading zeros (e.g. "075" → "75") on ~2,011 rows.
  Fixed by rebuilding `combo` from the zero-padded digit columns directly.

## 3. Methodology

All analysis follows one discipline throughout: **train on historical data,
predict/test against a genuinely held-out set the method never saw**,
exactly as Cardano's own reasoning about large numbers demands — small
samples produce noisy, misleading deviations, so every finding here is
checked against held-out data before being trusted.

### 3.1 Chi-square goodness-of-fit (`analysis/bias_test.py`)

For each (draw slot × digit position) pair, tests whether observed digit
frequencies (0-9) deviate from the uniform 10%-each expectation more than
sampling noise would explain. Includes Benjamini-Hochberg FDR correction
across the 9 simultaneous tests, since running many tests at once inflates
the chance of a false positive.

**Result: 0/9 significant, even before correction** (p-values 0.24-0.82).
No detectable digit bias in any slot or position.

### 3.2 Holdout prediction tests (`analysis/holdout_test.py`, `cold_number_test*.py`, `combined_*_test.py`)

Multiple prediction strategies, each trained only on data before a
held-out set of real draws, then scored on exact-combo-match accuracy:

- **Mode**: most frequent digit per position (1 guess).
- **Hot**: same idea, explicit "frequent digits repeat" strategy.
- **Cold**: the "hot numbers are due to regress" strategy (the gambler's
  fallacy, tested empirically rather than assumed).
- **Random**: seeded random baseline for comparison.

Tested under many variations: full dataset vs. per-slot isolation
(`cold_number_test_isolated.py`), pooled vs. isolated draw slots
(`combined_pooled_test.py`), August-only (`august_only_test.py`,
`august_only_pooled_test.py`), the single most-continuous slot across all
eras — 9PM (`combined_9pm_only_test.py`), 3 vs. 5 guesses per draw, and
holdout sizes from 3 to 10 draws.

**Result: consistently 0% (occasionally one incidental hit, always in the
opposite direction of the "cold numbers" hypothesis, and never
distinguishable from the random baseline).**

### 3.3 Layered/iterative slicing — a deliberate p-hacking demonstration (`analysis/phacking_demo.py`)

Tests whether repeatedly subdividing the data (by weekday, year, month,
quarter, etc.) and keeping whichever slice looks most "significant" can
manufacture a false-positive finding — a well-known statistical failure
mode (p-hacking / garden of forking paths), demonstrated concretely rather
than just asserted.

**Result:** slicing did surface one nominally "significant" p-value
(p=0.031, 9PM/digit-1/Saturdays) purely from trying 7 weekday
subsets — exactly the multiple-comparisons artifact predicted. Drilling
one layer deeper made it *less* significant (p=0.067) as the sample
shrank, and the "discovered rule" scored 0/3 against real held-out
draws — confirming it was noise, not signal. Extending to 10 candidate
slicing dimensions didn't change the outcome; the data ran out of usable
sample size (n<20) by layer 2-3 regardless.

### 3.4 Date-based grouping (`analysis/date_grouping_test.py`)

Tests whether restricting training data to the same day-of-month, the
same calendar date across years, or the same month produces any edge, on
the theory that seasonal/environmental factors (temperature, humidity)
might affect mechanical equipment.

**Result: 0% across same-day, same-day+month, and same-month groupings**,
with one incidental hot-strategy hit (again, opposite direction from the
cold-numbers hypothesis) attributable to ordinary variance at small n.

### 3.5 Neural network — honest vs. memorization (`analysis/nn_test.py`, `nn_tonight_guess.py`)

Two experiments run back-to-back specifically to make a subtle point
concrete:

- **Honest NN**: 3 small MLPClassifiers (date + lag features), trained
  only on ~7,944 real prior draws, tested on 10 genuinely unseen draws.
  **Result: 0/10** — same as every simpler method. More model complexity
  does not create information that isn't in the data.
- **"Cheating" NN**: identical architecture, but trained on *only* the 10
  holdout rows (over 1,300 parameters for 10 data points).
  **Result: 10/10, 100% — by memorization, not prediction.** Confirmed by
  inspecting the model directly: it assigns 98-99.8% confidence to each
  memorized answer, versus soft, uncertain probabilities when queried on
  a genuinely new date it was never shown the answer for. This
  experiment exists specifically to demonstrate what "training until it
  matches all the holdouts" actually requires — leaking the answer — and
  why that produces a number with zero forward-looking validity.

## 4. Conclusion

Across a chi-square uniformity test, ~15 prediction-strategy variations,
a deliberate p-hacking demonstration, seasonal/date-based slicing, and
both honest and memorization-based neural networks — using up to 20,200
historical draws spanning nearly 24 years — **no method found
statistically defensible, exploitable structure in Swertres draw
outcomes.** Every result is consistent with the draws being what they're
supposed to be: independent and uniformly random.

This doesn't retroactively prove the equipment-bias hypothesis was
unreasonable to test — it was a legitimate question, grounded in both
real-world precedent and centuries-old probability theory. It means that,
at the resolution this dataset allows, the answer is no.

## 5. Setup

```
python3 -m venv .venv
.venv/bin/pip install scipy pandas scikit-learn beautifulsoup4 lxml
```

Rebuild the data (optional — `data/*.csv` is already committed):

```
.venv/bin/python scraper/scrape.py 2022 2026
.venv/bin/python scraper/scrape_blogspot_2007_2012.py
.venv/bin/python scraper/scrape_gap_years.py
```

Run any analysis, e.g.:

```
.venv/bin/python analysis/bias_test.py
.venv/bin/python analysis/nn_test.py
```

## 6. Repository layout

```
data/           historical draw CSVs (date, draw_time, d1, d2, d3, combo)
scraper/        3 scrapers covering 2002-present across 2 source sites
analysis/       every experiment described above
docs/           supporting research notes (Cardano probability history)
```

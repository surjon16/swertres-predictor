"""
Chi-square goodness-of-fit test for digit bias in Swertres draws.

Hypothesis under test: mechanical/electronic draw equipment wear could
cause individual digit positions to deviate from a uniform 0-9
distribution. We test this per (draw_time, digit_position) bucket rather
than per full 3-digit combo, since 000-999 is too sparse (1000 possible
combos vs ~1100 draws per slot) for a reliable full-combo test.

H0: each digit position is uniformly distributed over 0-9 (fair/unbiased).
We reject H0 at alpha=0.05 if p < 0.05 for a given (slot, position) test.

Also runs a Benjamini-Hochberg FDR correction across all sub-tests, since
we're running 9 tests (3 slots x 3 positions) and would expect ~0.45
false positives at alpha=0.05 by chance alone.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history.csv"
DIGIT_COLS = ["d1", "d2", "d3"]
DRAW_SLOTS = ["2PM", "5PM", "9PM"]
ALPHA = 0.05


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    return df


def chi_square_uniform(observed_counts: pd.Series) -> tuple[float, float, int]:
    """Chi-square goodness-of-fit vs. uniform distribution over 0-9."""
    counts = observed_counts.reindex(range(10), fill_value=0)
    n = counts.sum()
    expected = n / 10
    chi2, p = stats.chisquare(counts, f_exp=[expected] * 10)
    return chi2, p, n


def benjamini_hochberg(pvalues: list[float], alpha: float = ALPHA) -> list[bool]:
    """Return a same-length list of bools: True if significant after FDR correction."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    thresholds = [(rank + 1) / m * alpha for rank in range(m)]
    significant = [False] * m
    max_rank_significant = -1
    for rank, idx in enumerate(order):
        if pvalues[idx] <= thresholds[rank]:
            max_rank_significant = rank
    if max_rank_significant >= 0:
        for rank in range(max_rank_significant + 1):
            significant[order[rank]] = True
    return significant


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slot in DRAW_SLOTS:
        slot_df = df[df["draw_time"] == slot]
        for pos, col in enumerate(DIGIT_COLS, start=1):
            counts = slot_df[col].value_counts()
            chi2, p, n = chi_square_uniform(counts)
            rows.append(
                {
                    "draw_slot": slot,
                    "digit_position": pos,
                    "n_draws": n,
                    "chi2": chi2,
                    "p_value": p,
                }
            )
    result = pd.DataFrame(rows)
    result["significant_raw"] = result["p_value"] < ALPHA
    result["significant_fdr"] = benjamini_hochberg(result["p_value"].tolist())
    return result


def digit_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Overall digit frequency across all positions/slots, for eyeballing skew direction."""
    all_digits = pd.concat([df[c] for c in DIGIT_COLS])
    freq = all_digits.value_counts().sort_index()
    pct = (freq / freq.sum() * 100).round(2)
    return pd.DataFrame({"digit": freq.index, "count": freq.values, "pct": pct.values})


def main():
    if not DATA_PATH.exists():
        print(f"No data at {DATA_PATH}. Run scraper/scrape.py first.", file=sys.stderr)
        sys.exit(1)

    df = load_data()
    print(f"Loaded {len(df)} draws from {DATA_PATH}\n")

    print("=== Overall digit frequency (all positions, all slots) ===")
    print(digit_frequency_table(df).to_string(index=False))
    print()

    print("=== Chi-square uniformity test: per draw-slot, per digit-position ===")
    print("H0: digit is uniform 0-9. alpha=0.05, FDR-corrected across 9 tests.\n")
    results = run_tests(df)
    print(results.to_string(index=False))
    print()

    n_sig_raw = results["significant_raw"].sum()
    n_sig_fdr = results["significant_fdr"].sum()
    print(f"Significant at raw alpha=0.05: {n_sig_raw}/9")
    print(f"Significant after FDR correction: {n_sig_fdr}/9")

    if n_sig_fdr == 0:
        print(
            "\nNo evidence of digit bias survives multiple-testing correction. "
            "With ~1,100 draws per slot, this rules out only moderate-or-larger "
            "biases -- a small skew could still be masked by sample size."
        )
    else:
        flagged = results[results["significant_fdr"]]
        print("\nFlagged (survives FDR correction) -- investigate further, do not "
              "treat as actionable on its own:")
        print(flagged.to_string(index=False))


if __name__ == "__main__":
    main()

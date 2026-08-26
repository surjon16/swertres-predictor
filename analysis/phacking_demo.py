"""
Demonstration: iterative/layered chi-square "slicing" on the TRAIN set only
(same holdout split as holdout_test.py) will eventually surface a
"significant" p-value purely from multiple comparisons -- even though the
data is random. Then we test whether that discovered pattern predicts the
real held-out draws. It should not do meaningfully better than chance,
because the "signal" was fit to noise in a specific historical slice.

Layers (each layer searches over candidate subsets of the CURRENT slice,
picks whichever subset has the smallest p-value, and drills into it):
  Layer 0:  full train set, by (draw_slot, digit_position)          [9 tests]
  Layer 1:  split further by weekday of the draw date                [x7]
  Layer 2:  split further by year                                    [x~5]
  Layer 3:  split further by month                                   [x12]
  Layer 4:  split further by odd/even day-of-month                   [x2]
  Layer 5:  split further by quarter                                 [x4]
  Layer 6:  split further by ISO week parity (odd/even week #)       [x2]
  Layer 7:  split further by half of month (1-15 vs 16-31)           [x2]
  Layer 8:  split further by weekend vs weekday                      [x2]
  Layer 9:  split further by semester (Jan-Jun vs Jul-Dec)           [x2]
  Layer 10: split further by week-of-month occurrence (1st..5th)     [x5]

In practice this will run out of usable sample size (our test refuses to
run chi-square on n<20) well before layer 10 -- that exhaustion is itself
part of the point: real signal should get MORE significant with more
data, not disappear because you ran out of rows to slice.

At each layer we keep whichever single sub-slice has the lowest p-value
so far and recurse into it -- this is deliberately how p-hacking works in
practice: "keep drilling into whatever looks most promising."
"""
from pathlib import Path

import pandas as pd
from scipy import stats

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history.csv"
HOLDOUT_N = 10
DIGIT_COLS = ["d1", "d2", "d3"]


def chi_square_uniform(counts: pd.Series):
    counts = counts.reindex(range(10), fill_value=0)
    n = counts.sum()
    if n < 20:
        return None  # too small to test meaningfully
    expected = n / 10
    chi2, p = stats.chisquare(counts, f_exp=[expected] * 10)
    return chi2, p, n


def best_split(df: pd.DataFrame, col: str, digit_col: str):
    """Try every value of `col` as a filter, return the one with lowest p-value."""
    best = None
    for val, sub in df.groupby(col):
        result = chi_square_uniform(sub[digit_col].value_counts())
        if result is None:
            continue
        chi2, p, n = result
        if best is None or p < best["p_value"]:
            best = {"filter_col": col, "filter_val": val, "chi2": chi2, "p_value": p, "n": n}
    return best


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "draw_time"]).reset_index(drop=True)

    train = df.iloc[:-HOLDOUT_N].copy()
    holdout = df.iloc[-HOLDOUT_N:].copy()

    train["weekday"] = train["date"].dt.day_name()
    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["day_parity"] = train["date"].dt.day % 2  # 0=even, 1=odd
    train["quarter"] = train["date"].dt.quarter
    train["iso_week_parity"] = train["date"].dt.isocalendar().week % 2
    train["half_of_month"] = (train["date"].dt.day > 15).astype(int)  # 0=1st half, 1=2nd half
    train["is_weekend"] = train["date"].dt.dayofweek.isin([5, 6]).astype(int)
    train["semester"] = (train["date"].dt.month > 6).astype(int)  # 0=Jan-Jun, 1=Jul-Dec
    train["week_of_month"] = ((train["date"].dt.day - 1) // 7) + 1  # 1st..5th occurrence of weekday

    layer_cols = [
        "weekday", "year", "month", "day_parity", "quarter",
        "iso_week_parity", "half_of_month", "is_weekend", "semester", "week_of_month",
    ]

    # Layer 0: find the single most "significant-looking" (slot, position) combo
    layer0 = []
    for slot in train["draw_time"].unique():
        for pos, col in enumerate(DIGIT_COLS, start=1):
            sub = train[train["draw_time"] == slot]
            result = chi_square_uniform(sub[col].value_counts())
            if result:
                chi2, p, n = result
                layer0.append({"draw_slot": slot, "digit_col": col, "chi2": chi2, "p_value": p, "n": n})
    layer0_df = pd.DataFrame(layer0).sort_values("p_value")
    print("=== Layer 0: full train set, all (slot, digit) combos ===")
    print(layer0_df.to_string(index=False))
    best0 = layer0_df.iloc[0]
    print(f"\nBest (lowest-p) starting point: slot={best0.draw_slot}, digit={best0.digit_col}, "
          f"p={best0.p_value:.4f}\n")

    current = train[train["draw_time"] == best0.draw_slot].copy()
    digit_col = best0.digit_col
    trail = [f"draw_time == {best0.draw_slot!r}"]
    p_trail = [best0.p_value]

    for layer_num, col in enumerate(layer_cols, start=1):
        result = best_split(current, col, digit_col)
        if result is None:
            print(f"Layer {layer_num}: no valid split on '{col}' (sample too small), stopping.")
            break
        current = current[current[col] == result["filter_val"]].copy()
        trail.append(f"{col} == {result['filter_val']!r}")
        p_trail.append(result["p_value"])
        print(f"=== Layer {layer_num}: drilling into {col} ===")
        print(f"  best sub-slice: {col}={result['filter_val']}, n={result['n']}, "
              f"chi2={result['chi2']:.2f}, p={result['p_value']:.4f}")
        print(f"  cumulative filter: {' AND '.join(trail)}\n")

    print("=== P-value trajectory across layers ===")
    for i, (t, p) in enumerate(zip(trail, p_trail)):
        print(f"  Layer {i}: p={p:.4f}  (filter so far: {' AND '.join(trail[:i+1])})")

    final_p = p_trail[-1]
    final_n = len(current)
    print(f"\nFinal slice: n={final_n} draws, p={final_p:.4f}, digit position tested: {digit_col}")
    if final_p < 0.05:
        print(">>> 'Significant' result found via layered slicing (as predicted: pure p-hacking artifact).")
    else:
        print(">>> Even after layering, no significance found in this run -- try more layers/columns.")

    # Derive the "discovered rule": most common digit in this narrow, cherry-picked slice
    discovered_digit = current[digit_col].value_counts().idxmax()
    print(f"\nDiscovered 'rule': in slice [{' AND '.join(trail)}], "
          f"digit '{discovered_digit}' is most common for {digit_col} "
          f"({(current[digit_col] == discovered_digit).mean()*100:.1f}% of the time in that slice).")

    # --- Now test this discovered rule against the REAL held-out draws ---
    print("\n=== Testing the discovered rule against the real held-out 10 draws ===")
    pos_index = DIGIT_COLS.index(digit_col)
    slot_filter = best0.draw_slot
    relevant_holdout = holdout[holdout["draw_time"] == slot_filter]
    print(f"(Rule only applies to draw_slot={slot_filter}; {len(relevant_holdout)} of the "
          f"10 holdout draws match that slot)")
    if len(relevant_holdout) == 0:
        print("No holdout draws in this slot to test against.")
    else:
        actual_digits = relevant_holdout[digit_col].tolist()
        hits = sum(d == discovered_digit for d in actual_digits)
        print(f"Actual {digit_col} values in holdout ({slot_filter}): {actual_digits}")
        print(f"Predicted digit (from p-hacked rule): {discovered_digit}")
        print(f"Hits: {hits}/{len(relevant_holdout)} "
              f"({hits/len(relevant_holdout)*100:.0f}% -- chance expectation is 10%)")


if __name__ == "__main__":
    main()

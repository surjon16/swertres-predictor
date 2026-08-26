"""
Final chi-square sweep: run goodness-of-fit + hot/cold/mode-derived guesses
across every reasonable training-set combination we've used in this
project, all aimed at the same target draw: 2026-08-26 9PM.

IMPORTANT: that draw's real result (075) is already recorded in
data/swertres_history_combined.csv (added in a prior session). Every
training subset below EXPLICITLY EXCLUDES that row, so this is a genuine
held-out test, not the model looking at its own answer. The true result
is only revealed at the very end for scoring.

"Every combination possible" = the groupings used throughout this
project's history:
  1. Full history, 9PM only
  2. Full history, all slots pooled (non-isolated)
  3. Same day-of-month only (26th of every month, all years)
  4. Same day-of-month AND month (Aug 26th across years)
  5. Same month only (all Augusts)
  6. Post-2022 era only (current 2PM/5PM/9PM schedule era)
  7. Pre-2022 combined (2002-2021, all prior eras)
  8. Weekday-matched (same day-of-week as 2026-08-26)
"""
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy import stats

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history_combined.csv"
DIGIT_COLS = ["d1", "d2", "d3"]
TARGET_DATE = pd.Timestamp("2026-08-26")
TARGET_SLOT = "9PM"
ACTUAL_RESULT = "075"  # revealed only for scoring at the end


def chi_square_uniform(counts: pd.Series):
    counts = counts.reindex(range(10), fill_value=0)
    n = counts.sum()
    if n < 20:
        return None
    expected = n / 10
    chi2, p = stats.chisquare(counts, f_exp=[expected] * 10)
    return chi2, p, n


def mode_combo(subset: pd.DataFrame) -> str:
    return "".join(str(Counter(subset[c]).most_common(1)[0][0]) for c in DIGIT_COLS)


def ranked_digits(subset: pd.DataFrame, col: str, ascending: bool) -> list[str]:
    counts = subset[col].value_counts().reindex(range(10), fill_value=0)
    return [str(d) for d in counts.sort_values(ascending=ascending).index]


def build_combos(subset: pd.DataFrame, ascending: bool, n: int) -> list[str]:
    ranked = {c: ranked_digits(subset, c, ascending) for c in DIGIT_COLS}
    return ["".join(ranked[c][r] for c in DIGIT_COLS) for r in range(n)]


def report(name: str, subset: pd.DataFrame, n_guesses: int = 3):
    if len(subset) < 20:
        print(f"{name}: n={len(subset)} -- too small to test, skipped")
        return None
    mode_g = mode_combo(subset)
    cold_g = build_combos(subset, ascending=True, n=n_guesses)
    hot_g = build_combos(subset, ascending=False, n=n_guesses)

    chi_results = []
    for col in DIGIT_COLS:
        r = chi_square_uniform(subset[col].value_counts())
        chi_results.append(r)
    avg_p = sum(r[1] for r in chi_results) / 3

    print(f"{name}  (n={len(subset)}, avg chi-square p={avg_p:.3f})")
    print(f"  mode={mode_g}  cold={cold_g}  hot={hot_g}")

    all_guesses = {mode_g, *cold_g, *hot_g}
    hit = ACTUAL_RESULT in all_guesses
    print(f"  -> {'HIT' if hit else 'miss'} (actual was {ACTUAL_RESULT})")
    return {"name": name, "n": len(subset), "avg_p": avg_p, "guesses": all_guesses, "hit": hit}


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": str, "d2": str, "d3": str})
    df["combo"] = df["d1"] + df["d2"] + df["d3"]
    for c in DIGIT_COLS:
        df[c] = df[c].astype(int)
    df["date"] = pd.to_datetime(df["date"])

    # Exclude the target draw itself from ALL training subsets -- true holdout
    is_target_row = (df["date"] == TARGET_DATE) & (df["draw_time"] == TARGET_SLOT)
    assert is_target_row.sum() == 1, "expected exactly one target row"
    print(f"Excluding the actual target row from all training data (true holdout):")
    print(df[is_target_row].to_string(index=False))
    print()

    train_all = df[~is_target_row]
    nine = train_all[train_all["draw_time"] == TARGET_SLOT]

    combos = {}

    combos["1. Full history, 9PM only"] = nine
    combos["2. Full history, all slots pooled"] = train_all
    combos["3. Same day-of-month (26th, all months/years, 9PM)"] = nine[nine["date"].dt.day == 26]
    combos["4. Same day+month (Aug 26th across years, 9PM)"] = nine[
        (nine["date"].dt.day == 26) & (nine["date"].dt.month == 8)
    ]
    combos["5. Same month only (all Augusts, 9PM)"] = nine[nine["date"].dt.month == 8]
    combos["6. Post-2022 era only (9PM)"] = nine[nine["date"].dt.year >= 2022]
    combos["7. Pre-2022, all prior eras combined (9PM)"] = nine[nine["date"].dt.year < 2022]
    combos["8. Same weekday as target (9PM)"] = nine[nine["date"].dt.dayofweek == TARGET_DATE.dayofweek]

    results = []
    for name, subset in combos.items():
        r = report(name, subset, n_guesses=3)
        if r:
            results.append(r)
        print()

    print("=" * 60)
    print(f"REVEAL: actual 2026-08-26 9PM result = {ACTUAL_RESULT}")
    print("=" * 60)
    n_hits = sum(r["hit"] for r in results)
    print(f"Combinations tested: {len(results)}")
    print(f"Combinations that included {ACTUAL_RESULT} among their guesses: {n_hits}")
    for r in results:
        print(f"  [{'HIT' if r['hit'] else 'miss'}] {r['name']} (n={r['n']}, p={r['avg_p']:.3f})")


if __name__ == "__main__":
    main()

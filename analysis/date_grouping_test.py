"""
Tests three date-based training-set filters against the real held-out
last 10 draws, using the 9PM slot only (the one continuous process across
the full 2002-2026 history). For EACH held-out draw, the training subset
is built dynamically based on that draw's own date:

  1. same_day:       all historical 9PM draws where day-of-month matches
                      (e.g. predicting an Aug-25 draw uses every historical
                      25th -- Jan 25, Feb 25, ..., Dec 25 -- across all years)
  2. same_day_month: historical 9PM draws matching BOTH day-of-month AND
                      month (e.g. only prior Aug-25ths -- the exact
                      calendar date across years)
  3. same_month:     historical 9PM draws in the same month, any day
                      (e.g. all prior Augusts)

For each strategy, mode/cold/hot combos are derived from that draw-specific
training subset (methods from mode_combo/cold_number_test). A single
random baseline (not date-dependent) is included for comparison.

Sample sizes vary a lot by strategy -- same_day_month in particular may
have very few historical examples (one draw's-worth per year at most) --
so subset sizes are reported alongside hit rates.
"""
import random
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history_combined.csv"
HOLDOUT_N = 10
DIGIT_COLS = ["d1", "d2", "d3"]
N_GUESSES = 3


def ranked_digits(subset: pd.DataFrame, col: str, ascending: bool) -> list[str]:
    counts = subset[col].value_counts().reindex(range(10), fill_value=0)
    ordered = counts.sort_values(ascending=ascending)
    return [str(d) for d in ordered.index]


def build_combos(subset: pd.DataFrame, ascending: bool, n: int) -> list[str]:
    ranked = {col: ranked_digits(subset, col, ascending) for col in DIGIT_COLS}
    return ["".join(ranked[col][r] for col in DIGIT_COLS) for r in range(n)]


def mode_combo(subset: pd.DataFrame) -> str:
    return "".join(str(Counter(subset[col]).most_common(1)[0][0]) for col in DIGIT_COLS)


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df["combo"] = df["combo"].astype(str).str.zfill(3)
    df["date"] = pd.to_datetime(df["date"])

    nine = df[df["draw_time"] == "9PM"].sort_values("date").reset_index(drop=True)
    train_all = nine.iloc[:-HOLDOUT_N]
    holdout = nine.iloc[-HOLDOUT_N:]

    print(f"9PM-only: {len(nine)} draws total, holdout = last {HOLDOUT_N}\n")

    rng = random.Random(42)
    random_combos = [f"{rng.randint(0,9)}{rng.randint(0,9)}{rng.randint(0,9)}" for _ in range(N_GUESSES)]

    strategies = {
        "same_day": lambda d: train_all[train_all["date"].dt.day == d.day],
        "same_day_month": lambda d: train_all[
            (train_all["date"].dt.day == d.day) & (train_all["date"].dt.month == d.month)
        ],
        "same_month": lambda d: train_all[train_all["date"].dt.month == d.month],
    }

    hit_counts = {f"{s}_mode": 0 for s in strategies}
    hit_counts.update({f"{s}_cold": 0 for s in strategies})
    hit_counts.update({f"{s}_hot": 0 for s in strategies})
    hit_counts["random"] = 0

    detail_rows = []
    for row in holdout.itertuples():
        d = row.date
        rec = {"date": d.date(), "actual": row.combo}
        for strat_name, filt in strategies.items():
            subset = filt(d)
            n = len(subset)
            if n < 10:
                rec[f"{strat_name}_n"] = n
                rec[f"{strat_name}_mode_hit"] = None
                rec[f"{strat_name}_cold_hit"] = None
                rec[f"{strat_name}_hot_hit"] = None
                continue
            mode_g = mode_combo(subset)
            cold_g = build_combos(subset, ascending=True, n=N_GUESSES)
            hot_g = build_combos(subset, ascending=False, n=N_GUESSES)
            rec[f"{strat_name}_n"] = n
            rec[f"{strat_name}_mode_hit"] = row.combo == mode_g
            rec[f"{strat_name}_cold_hit"] = row.combo in cold_g
            rec[f"{strat_name}_hot_hit"] = row.combo in hot_g
            hit_counts[f"{strat_name}_mode"] += int(rec[f"{strat_name}_mode_hit"])
            hit_counts[f"{strat_name}_cold"] += int(rec[f"{strat_name}_cold_hit"])
            hit_counts[f"{strat_name}_hot"] += int(rec[f"{strat_name}_hot_hit"])
        rec["random_hit"] = row.combo in random_combos
        hit_counts["random"] += int(rec["random_hit"])
        detail_rows.append(rec)

    detail_df = pd.DataFrame(detail_rows)
    print(detail_df.to_string(index=False))
    print()

    print("=== Summary: hits / 10 (mode=1 guess, cold/hot=3 guesses, random=3 guesses) ===")
    for strat in strategies:
        avg_n = detail_df[f"{strat}_n"].mean()
        print(f"\n{strat} (avg training subset size: {avg_n:.0f}):")
        for method in ["mode", "cold", "hot"]:
            n_guesses = 1 if method == "mode" else N_GUESSES
            expected_pct = n_guesses / 1000 * 100
            hits = hit_counts[f"{strat}_{method}"]
            print(f"  {method}: {hits}/{HOLDOUT_N} ({hits/HOLDOUT_N*100:.0f}%, chance={expected_pct:.1f}%)")
    print(f"\nrandom (shared baseline, not date-dependent): {hit_counts['random']}/{HOLDOUT_N} "
          f"({hit_counts['random']/HOLDOUT_N*100:.0f}%, chance={N_GUESSES/1000*100:.1f}%)")


if __name__ == "__main__":
    main()

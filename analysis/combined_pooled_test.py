"""
Runs all prediction methods built so far (mode predictor, cold/hot/random
3-guess) against the FULL combined dataset (2002-2026, ~20,200 draws),
WITHOUT slot isolation -- all draw_time labels pooled together for both
training and the holdout, same as the original (pre-isolation)
holdout_test.py / cold_number_test.py.

Note: pooling mixes eras with different actual draw schedules (9PM-only
2002-2006, 11AM/4PM/9PM 2011-2020, etc.) -- this is what "non-isolated"
means here: no separation by slot OR by era, all draws treated as one
undifferentiated pool.
"""
import random
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history_combined.csv"
HOLDOUT_N = 10
DIGIT_COLS = ["d1", "d2", "d3"]
N_GUESSES = 3


def ranked_digits(train: pd.DataFrame, col: str, ascending: bool) -> list[str]:
    counts = train[col].value_counts().reindex(range(10), fill_value=0)
    ordered = counts.sort_values(ascending=ascending)
    return [str(d) for d in ordered.index]


def build_combos(train: pd.DataFrame, ascending: bool, n: int) -> list[str]:
    ranked = {col: ranked_digits(train, col, ascending) for col in DIGIT_COLS}
    return ["".join(ranked[col][r] for col in DIGIT_COLS) for r in range(n)]


def mode_combo(train: pd.DataFrame) -> str:
    return "".join(str(Counter(train[col]).most_common(1)[0][0]) for col in DIGIT_COLS)


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df["combo"] = df["combo"].astype(str).str.zfill(3)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "draw_time"]).reset_index(drop=True)

    print(f"Full combined dataset: {len(df)} draws, {df['date'].min().date()} -> {df['date'].max().date()}\n")

    train = df.iloc[:-HOLDOUT_N]
    holdout = df.iloc[-HOLDOUT_N:]
    print(f"Train n={len(train)}, holdout n={len(holdout)} (all slots/eras pooled in both)\n")

    rng = random.Random(42)
    mode_guess = mode_combo(train)
    cold_combos = build_combos(train, ascending=True, n=N_GUESSES)
    hot_combos = build_combos(train, ascending=False, n=N_GUESSES)
    random_combos = [f"{rng.randint(0,9)}{rng.randint(0,9)}{rng.randint(0,9)}" for _ in range(N_GUESSES)]

    print(f"mode={mode_guess}")
    print(f"cold={cold_combos}")
    print(f"hot={hot_combos}")
    print(f"random={random_combos}\n")

    rows = []
    hit_counts = {"mode": 0, "cold": 0, "hot": 0, "random": 0}
    for row in holdout.itertuples():
        record = {"date": row.date.date(), "draw_time": row.draw_time, "actual": row.combo}
        record["mode_hit"] = row.combo == mode_guess
        hit_counts["mode"] += int(record["mode_hit"])
        for strat_name, combos in [("cold", cold_combos), ("hot", hot_combos), ("random", random_combos)]:
            hit = row.combo in combos
            record[f"{strat_name}_hit"] = hit
            hit_counts[strat_name] += int(hit)
        rows.append(record)

    print(pd.DataFrame(rows).to_string(index=False))
    print()
    for strat in ["mode", "cold", "hot", "random"]:
        n_guesses = 1 if strat == "mode" else N_GUESSES
        expected_pct = n_guesses / 1000 * 100
        print(f"  {strat}: {hit_counts[strat]}/{HOLDOUT_N} hits "
              f"({hit_counts[strat]/HOLDOUT_N*100:.0f}%, chance={expected_pct:.1f}%)")


if __name__ == "__main__":
    main()

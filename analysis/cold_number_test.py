"""
Tests the "cold number" strategy (avoid recently/historically frequent
digits, favor rare ones) against the real held-out 10 draws, with 3
candidate combos per draw instead of 1. Compared against a "hot number"
strategy (favor frequent digits) and pure random, also 3 combos each, as
controls.

If cold-picking has any real edge, it should beat both hot-picking and
random by a wide margin here. If draws are independent (which our
chi-square test found no evidence against), all three should perform
about the same: ~3/1000 chance of an exact hit per draw with 3 guesses,
i.e. ~0.3 expected hits total across 10 draws for any of the three
strategies.
"""
import random
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history.csv"
HOLDOUT_N = 10
DIGIT_COLS = ["d1", "d2", "d3"]
N_GUESSES = 3


def ranked_digits(train_slot: pd.DataFrame, col: str, ascending: bool) -> list[str]:
    counts = train_slot[col].value_counts().reindex(range(10), fill_value=0)
    ordered = counts.sort_values(ascending=ascending)
    return [str(d) for d in ordered.index]


def build_combos(train_slot: pd.DataFrame, ascending: bool, n: int) -> list[str]:
    """n combos, rank r built from the r-th coldest/hottest digit in each position."""
    ranked = {col: ranked_digits(train_slot, col, ascending) for col in DIGIT_COLS}
    return ["".join(ranked[col][r] for col in DIGIT_COLS) for r in range(n)]


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df["combo"] = df["combo"].astype(str).str.zfill(3)
    df = df.sort_values(["date", "draw_time"]).reset_index(drop=True)

    train = df.iloc[:-HOLDOUT_N].copy()
    holdout = df.iloc[-HOLDOUT_N:].copy()

    rng = random.Random(42)

    strategies = {}
    for slot in train["draw_time"].unique():
        slot_train = train[train["draw_time"] == slot]
        cold_combos = build_combos(slot_train, ascending=True, n=N_GUESSES)
        hot_combos = build_combos(slot_train, ascending=False, n=N_GUESSES)
        random_combos = [
            f"{rng.randint(0,9)}{rng.randint(0,9)}{rng.randint(0,9)}" for _ in range(N_GUESSES)
        ]
        strategies[slot] = {"cold": cold_combos, "hot": hot_combos, "random": random_combos}

    print("Per-slot 3-combo guesses derived from train set only:\n")
    for slot, s in strategies.items():
        print(f"  {slot}: cold={s['cold']}  hot={s['hot']}  random={s['random']}")
    print()

    rows = []
    hit_counts = {"cold": 0, "hot": 0, "random": 0}
    for row in holdout.itertuples():
        slot_strats = strategies[row.draw_time]
        record = {"date": row.date, "draw_time": row.draw_time, "actual": row.combo}
        for strat_name, combos in slot_strats.items():
            hit = row.combo in combos
            record[f"{strat_name}_hit"] = hit
            hit_counts[strat_name] += int(hit)
        rows.append(record)

    results = pd.DataFrame(rows)
    print("=== Results on the real held-out 10 draws (3 guesses per strategy per draw) ===")
    print(results.to_string(index=False))
    print()

    for strat in ["cold", "hot", "random"]:
        pct = hit_counts[strat] / HOLDOUT_N * 100
        print(f"{strat.capitalize()}-number strategy: {hit_counts[strat]}/{HOLDOUT_N} hits ({pct:.0f}%)")

    print(f"\nExpected by chance alone with 3 guesses/draw: ~0.3 hits total across 10 draws (3%)")


if __name__ == "__main__":
    main()

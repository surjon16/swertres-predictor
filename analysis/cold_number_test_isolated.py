"""
Same cold/hot/random 3-combo test as cold_number_test.py, but fully
isolated per draw slot: each slot (2PM, 5PM, 9PM) gets its own holdout of
its last 10 draws, trained only on that slot's own prior history. No
cross-slot mixing anywhere -- 2PM never sees 5PM or 9PM data, etc.

This also fixes an issue in the original test: holding out "the last 10
rows of the combined file" split unevenly across slots (3/4/3). Here each
slot gets a clean, equal 10-draw holdout.
"""
import random
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history.csv"
HOLDOUT_N = 10
DIGIT_COLS = ["d1", "d2", "d3"]
N_GUESSES = 3
SLOTS = ["2PM", "5PM", "9PM"]


def ranked_digits(train_slot: pd.DataFrame, col: str, ascending: bool) -> list[str]:
    counts = train_slot[col].value_counts().reindex(range(10), fill_value=0)
    ordered = counts.sort_values(ascending=ascending)
    return [str(d) for d in ordered.index]


def build_combos(train_slot: pd.DataFrame, ascending: bool, n: int) -> list[str]:
    ranked = {col: ranked_digits(train_slot, col, ascending) for col in DIGIT_COLS}
    return ["".join(ranked[col][r] for col in DIGIT_COLS) for r in range(n)]


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df["combo"] = df["combo"].astype(str).str.zfill(3)
    df = df.sort_values(["date", "draw_time"]).reset_index(drop=True)

    rng = random.Random(42)
    overall_hits = {"cold": 0, "hot": 0, "random": 0}
    overall_n = 0

    for slot in SLOTS:
        slot_df = df[df["draw_time"] == slot].sort_values("date").reset_index(drop=True)
        train = slot_df.iloc[:-HOLDOUT_N]
        holdout = slot_df.iloc[-HOLDOUT_N:]

        print(f"\n{'='*60}")
        print(f"Slot: {slot}  (train n={len(train)}, holdout n={len(holdout)}, "
              f"isolated -- no other-slot data used)")
        print(f"{'='*60}")

        cold_combos = build_combos(train, ascending=True, n=N_GUESSES)
        hot_combos = build_combos(train, ascending=False, n=N_GUESSES)
        random_combos = [
            f"{rng.randint(0,9)}{rng.randint(0,9)}{rng.randint(0,9)}" for _ in range(N_GUESSES)
        ]
        print(f"cold={cold_combos}  hot={hot_combos}  random={random_combos}\n")

        rows = []
        hit_counts = {"cold": 0, "hot": 0, "random": 0}
        for row in holdout.itertuples():
            record = {"date": row.date, "actual": row.combo}
            for strat_name, combos in [("cold", cold_combos), ("hot", hot_combos), ("random", random_combos)]:
                hit = row.combo in combos
                record[f"{strat_name}_hit"] = hit
                hit_counts[strat_name] += int(hit)
                overall_hits[strat_name] += int(hit)
            rows.append(record)
        overall_n += len(holdout)

        print(pd.DataFrame(rows).to_string(index=False))
        print()
        for strat in ["cold", "hot", "random"]:
            print(f"  {strat}: {hit_counts[strat]}/{HOLDOUT_N} hits "
                  f"({hit_counts[strat]/HOLDOUT_N*100:.0f}%)")

    print(f"\n{'='*60}")
    print(f"TOTAL across all 3 isolated slots (n={overall_n} draws, {N_GUESSES} guesses each)")
    print(f"{'='*60}")
    for strat in ["cold", "hot", "random"]:
        pct = overall_hits[strat] / overall_n * 100
        print(f"  {strat}: {overall_hits[strat]}/{overall_n} hits ({pct:.1f}%)")
    print(f"  Expected by chance alone: ~{overall_n * N_GUESSES / 1000:.2f} hits "
          f"({overall_n * N_GUESSES / 1000 / overall_n * 100:.1f}%)")


if __name__ == "__main__":
    main()

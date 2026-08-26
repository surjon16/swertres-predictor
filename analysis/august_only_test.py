"""
Restricts the dataset to August draws only (all years: 2022-2026, ~142
draws per slot), isolated per draw slot (2PM/5PM/9PM never mix), and runs
every prediction method built so far against a held-out set of the most
recent August draws per slot:

  1. Mode predictor: single combo = most frequent digit per position
     (from holdout_test.py).
  2. Cold / Hot / Random 3-guess strategy (from cold_number_test_isolated.py).

Rationale for restricting to August: if there's some seasonal effect
(temperature/humidity affecting mechanical equipment, etc.), it might
only show up within a single month's draws pooled across years -- this
tests that specific hypothesis rather than the full-year pool.
"""
import random
from collections import Counter
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


def mode_combo(train_slot: pd.DataFrame) -> str:
    return "".join(str(Counter(train_slot[col]).most_common(1)[0][0]) for col in DIGIT_COLS)


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df["combo"] = df["combo"].astype(str).str.zfill(3)
    df["date"] = pd.to_datetime(df["date"])

    august = df[df["date"].dt.month == 8].copy()
    print(f"Total August draws (all years, all slots): {len(august)}\n")

    rng = random.Random(42)
    overall = {"mode": 0, "cold": 0, "hot": 0, "random": 0}
    overall_n = 0

    for slot in SLOTS:
        slot_df = august[august["draw_time"] == slot].sort_values("date").reset_index(drop=True)
        if len(slot_df) <= HOLDOUT_N:
            print(f"Slot {slot}: not enough August draws ({len(slot_df)}), skipping.")
            continue

        train = slot_df.iloc[:-HOLDOUT_N]
        holdout = slot_df.iloc[-HOLDOUT_N:]

        print(f"{'='*60}")
        print(f"Slot: {slot} (August only, isolated) -- train n={len(train)}, holdout n={len(holdout)}")
        print(f"{'='*60}")

        mode_guess = mode_combo(train)
        cold_combos = build_combos(train, ascending=True, n=N_GUESSES)
        hot_combos = build_combos(train, ascending=False, n=N_GUESSES)
        random_combos = [
            f"{rng.randint(0,9)}{rng.randint(0,9)}{rng.randint(0,9)}" for _ in range(N_GUESSES)
        ]

        print(f"mode={mode_guess}  cold={cold_combos}  hot={hot_combos}  random={random_combos}\n")

        rows = []
        hit_counts = {"mode": 0, "cold": 0, "hot": 0, "random": 0}
        for row in holdout.itertuples():
            record = {"date": row.date.date(), "actual": row.combo}
            record["mode_hit"] = row.combo == mode_guess
            hit_counts["mode"] += int(record["mode_hit"])
            for strat_name, combos in [("cold", cold_combos), ("hot", hot_combos), ("random", random_combos)]:
                hit = row.combo in combos
                record[f"{strat_name}_hit"] = hit
                hit_counts[strat_name] += int(hit)
            rows.append(record)
            overall_n += 1
            for k in overall:
                overall[k] += int(record.get(f"{k}_hit", False))

        print(pd.DataFrame(rows).to_string(index=False))
        print()
        for strat in ["mode", "cold", "hot", "random"]:
            n_guesses = 1 if strat == "mode" else N_GUESSES
            expected_pct = n_guesses / 1000 * 100
            print(f"  {strat}: {hit_counts[strat]}/{HOLDOUT_N} hits "
                  f"({hit_counts[strat]/HOLDOUT_N*100:.0f}%, chance={expected_pct:.1f}%)")
        print()

    print(f"{'='*60}")
    print(f"TOTAL across all isolated slots (August only, n={overall_n} holdout draws)")
    print(f"{'='*60}")
    for strat in ["mode", "cold", "hot", "random"]:
        n_guesses = 1 if strat == "mode" else N_GUESSES
        expected = overall_n * n_guesses / 1000
        pct = overall[strat] / overall_n * 100 if overall_n else 0
        print(f"  {strat}: {overall[strat]}/{overall_n} hits ({pct:.1f}%), expected by chance: {expected:.2f}")


if __name__ == "__main__":
    main()

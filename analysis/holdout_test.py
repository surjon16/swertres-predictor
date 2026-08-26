"""
Honest holdout test: hold out the last 10 draws, train the best legitimate
predictor we can on everything before them, predict the 10 held-out draws,
and report the REAL exact-match accuracy. No peeking at the holdout set
during model fitting.

Two predictors are compared:
  1. "Most frequent digit per position" (per draw-slot) -- the strongest
     signal a frequency-based method could possibly use.
  2. Pure random guess, as a sanity-check baseline (should perform
     similarly to #1 if the draws are truly uniform random, which the
     prior chi-square test found no evidence against).

This will NOT hit 90% exact-match accuracy if the draws are fair, and
that outcome is reported as-is, not adjusted.
"""
import random
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history.csv"
HOLDOUT_N = 10
DIGIT_COLS = ["d1", "d2", "d3"]


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df = df.sort_values(["date", "draw_time"]).reset_index(drop=True)

    train = df.iloc[:-HOLDOUT_N].copy()
    holdout = df.iloc[-HOLDOUT_N:].copy()

    print(f"Train set: {len(train)} draws")
    print(f"Holdout set: {len(holdout)} draws (the actual last 10)\n")
    print("Holdout draws (ground truth):")
    print(holdout[["date", "draw_time", "combo"]].to_string(index=False))
    print()

    # --- Predictor 1: most-frequent-digit-per-position, per draw slot ---
    mode_by_slot = {}
    for slot in train["draw_time"].unique():
        slot_train = train[train["draw_time"] == slot]
        mode_digits = []
        for col in DIGIT_COLS:
            most_common = Counter(slot_train[col]).most_common(1)[0][0]
            mode_digits.append(str(most_common))
        mode_by_slot[slot] = "".join(mode_digits)

    print("Predictor 1 (mode digit per position, per slot), trained on train set only:")
    for slot, combo in mode_by_slot.items():
        print(f"  {slot}: always predict {combo}")
    print()

    freq_predictions = [mode_by_slot[row.draw_time] for row in holdout.itertuples()]
    freq_hits = sum(
        pred == actual for pred, actual in zip(freq_predictions, holdout["combo"].astype(str).str.zfill(3))
    )

    # --- Predictor 2: pure random guess baseline (seeded for reproducibility) ---
    rng = random.Random(42)
    random_predictions = [f"{rng.randint(0,9)}{rng.randint(0,9)}{rng.randint(0,9)}" for _ in range(HOLDOUT_N)]
    random_hits = sum(
        pred == actual for pred, actual in zip(random_predictions, holdout["combo"].astype(str).str.zfill(3))
    )

    results = holdout[["date", "draw_time", "combo"]].copy()
    results["combo"] = results["combo"].astype(str).str.zfill(3)
    results["freq_predictor_guess"] = freq_predictions
    results["freq_predictor_hit"] = results["combo"] == results["freq_predictor_guess"]
    results["random_guess"] = random_predictions
    results["random_hit"] = results["combo"] == results["random_guess"]

    print("=== Results on the real held-out 10 draws ===")
    print(results.to_string(index=False))
    print()
    print(f"Frequency-based predictor: {freq_hits}/{HOLDOUT_N} exact matches "
          f"({freq_hits/HOLDOUT_N*100:.0f}% accuracy)")
    print(f"Pure random guess:         {random_hits}/{HOLDOUT_N} exact matches "
          f"({random_hits/HOLDOUT_N*100:.0f}% accuracy)")
    print(f"Expected by chance alone:  ~0.1 exact matches per 10 draws (1/1000 per draw)")


if __name__ == "__main__":
    main()

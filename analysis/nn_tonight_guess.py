"""
Reuses the "cheating" NN from nn_test.py (memorized on the 10 known 9PM
draws 2026-08-16 through 2026-08-25) to produce guesses for a date it was
NOT trained on: the actual next/upcoming draw.

IMPORTANT CAVEAT: the cheating NN's 10/10 result in nn_test.py worked
because it was trained directly on the answers it was then asked to
"predict" -- pure memorization, not forecasting. For a draw that hasn't
happened yet, there is no answer to memorize, so this script is really
just running the SAME memorized model on an out-of-sample date and
watching it interpolate between the 10 points it knows. The soft,
non-confident output probabilities (compare to the 98%+ confidences on
its actual memorized training rows) demonstrate this directly -- it is
not a working predictor, and any digits it outputs carry no more
information than a random guess about a draw that hasn't been decided
yet.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_test import build_features, DIGIT_COLS, HOLDOUT_N, train_predict, top_k_combos  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history_combined.csv"
N_GUESSES = 5


def main(target_date_str: str):
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df["combo"] = df["combo"].astype(str).str.zfill(3)
    df["date"] = pd.to_datetime(df["date"])
    nine = df[df["draw_time"] == "9PM"].sort_values("date").reset_index(drop=True)

    feats = build_features(nine).dropna()
    labels = nine.loc[feats.index]
    holdout_feats = feats.iloc[-HOLDOUT_N:]
    holdout_labels = labels.iloc[-HOLDOUT_N:]

    print("Model memorized on these 10 known draws:")
    print(holdout_labels[["date", "combo"]].to_string(index=False))
    print()

    target_date = pd.Timestamp(target_date_str)
    last5 = (
        nine[nine["date"] < target_date]
        .sort_values("date")
        .tail(5)
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    print(f"Most recent known draws used for {target_date.date()}'s lag features:")
    print(last5[["date", "combo"]].to_string(index=False))

    row = {
        "dow": target_date.dayofweek,
        "month": target_date.month,
        "year": target_date.year,
        "doy_sin": np.sin(2 * np.pi * target_date.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * target_date.dayofyear / 365.25),
    }
    for lag in range(1, 6):
        r = last5.iloc[lag - 1]
        for col in DIGIT_COLS:
            row[f"lag{lag}_{col}"] = r[col]
    target_feats = pd.DataFrame([row])[holdout_feats.columns]

    probs = {col: train_predict(holdout_feats, holdout_labels, target_feats, col) for col in DIGIT_COLS}
    guesses = top_k_combos({col: probs[col][0] for col in DIGIT_COLS}, N_GUESSES)

    print(f"\n'Cheating' NN guesses for {target_date.date()} 9PM (out-of-sample, NOT memorized): {guesses}")
    for col in DIGIT_COLS:
        print(f"  {col} distribution: {np.round(probs[col][0], 3)}")
    print(
        "\nNote the soft/uncertain probabilities here vs. the 98%+ confidence on rows it "
        "actually memorized (see nn_test.py output) -- this is interpolation between "
        "memorized points, not a real prediction. Treat these guesses as no better than random."
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "2026-08-26"
    main(target)

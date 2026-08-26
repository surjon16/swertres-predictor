"""
Two neural-network experiments on the 9PM slot (2002-2026, ~8k draws),
run back to back so the difference between them is visible directly.

EXPERIMENT 1 -- "honest" NN: 3 small MLPClassifiers (one per digit
position) trained ONLY on the training set (everything except the real
last-10 holdout), using date features (day-of-week/month/year, cyclical
day-of-year) plus lag features from the 5 preceding real draws. Predicts
probabilities for the 10 held-out draws it has never seen. Top-5 highest
joint-probability combos are taken as its 5 guesses per draw, scored for
exact match -- same protocol as every other method in this repo.

EXPERIMENT 2 -- "cheating" NN: identical architecture, but trained with
the holdout labels included in the training data (i.e. it directly sees
the answers before being asked to predict them). This is included ONLY to
demonstrate that a NN CAN trivially "match all 10 holdouts" if allowed to
memorize them -- that says nothing about predictive power on a fair
random process, it's the digital equivalent of writing down the answer
key. Do not mistake this for a working predictor.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swertres_history_combined.csv"
HOLDOUT_N = 10
DIGIT_COLS = ["d1", "d2", "d3"]
N_LAGS = 5
N_GUESSES = 5
RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["dow"] = df["date"].dt.dayofweek
    feats["month"] = df["date"].dt.month
    feats["year"] = df["date"].dt.year
    doy = df["date"].dt.dayofyear
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for lag in range(1, N_LAGS + 1):
        for col in DIGIT_COLS:
            feats[f"lag{lag}_{col}"] = df[col].shift(lag)
    return feats


def top_k_combos(probs: dict[str, np.ndarray], k: int) -> list[str]:
    """probs: {'d1': array[10], 'd2': array[10], 'd3': array[10]} -> top-k combos by joint prob."""
    joint = []
    for a in range(10):
        for b in range(10):
            for c in range(10):
                p = probs["d1"][a] * probs["d2"][b] * probs["d3"][c]
                joint.append((p, f"{a}{b}{c}"))
    joint.sort(reverse=True)
    return [combo for _, combo in joint[:k]]


def train_predict(train_feats, train_labels, holdout_feats, digit_col):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats)
    X_holdout = scaler.transform(holdout_feats)
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )
    clf.fit(X_train, train_labels[digit_col])
    proba = clf.predict_proba(X_holdout)
    # align columns to digits 0-9 (classifier may not have seen all classes)
    full = np.zeros((len(X_holdout), 10))
    for i, cls in enumerate(clf.classes_):
        full[:, cls] = proba[:, i]
    return full


def main():
    df = pd.read_csv(DATA_PATH, dtype={"d1": int, "d2": int, "d3": int})
    df["combo"] = df["combo"].astype(str).str.zfill(3)
    df["date"] = pd.to_datetime(df["date"])
    nine = df[df["draw_time"] == "9PM"].sort_values("date").reset_index(drop=True)

    feats = build_features(nine)
    valid = feats.dropna().index  # drop first N_LAGS rows (no lag history yet)
    feats = feats.loc[valid]
    labels = nine.loc[valid]

    train_feats = feats.iloc[:-HOLDOUT_N]
    train_labels = labels.iloc[:-HOLDOUT_N]
    holdout_feats = feats.iloc[-HOLDOUT_N:]
    holdout_labels = labels.iloc[-HOLDOUT_N:]

    print(f"9PM draws usable: {len(feats)} (after dropping {N_LAGS} for lag warmup)")
    print(f"Train n={len(train_feats)}, holdout n={len(holdout_feats)}\n")

    # ---------- Experiment 1: honest ----------
    print("=" * 60)
    print("EXPERIMENT 1: Honest NN (trained ONLY on data before the holdout)")
    print("=" * 60)
    honest_hits = 0
    rows = []
    for digit_col in DIGIT_COLS:
        pass  # trained inside loop below per draw set jointly
    probs_by_digit = {
        col: train_predict(train_feats, train_labels, holdout_feats, col) for col in DIGIT_COLS
    }
    for i, row in enumerate(holdout_labels.itertuples()):
        probs = {col: probs_by_digit[col][i] for col in DIGIT_COLS}
        guesses = top_k_combos(probs, N_GUESSES)
        hit = row.combo in guesses
        honest_hits += int(hit)
        rows.append({"date": row.date.date(), "actual": row.combo, "guesses": guesses, "hit": hit})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nHonest NN: {honest_hits}/{HOLDOUT_N} exact matches "
          f"({honest_hits/HOLDOUT_N*100:.0f}%, chance with {N_GUESSES} guesses = {N_GUESSES/1000*100:.1f}%)")

    # ---------- Experiment 2: cheating (trained to memorize the holdout directly) ----------
    # Note: training WITH the 10 holdout rows mixed into the 7944 real training
    # rows (as tried first) barely changes anything -- 10 examples carry
    # negligible gradient weight among 7944, so the network doesn't actually
    # memorize them (see honest-vs-mixed-in result, both ~0%). To actually
    # demonstrate memorization, the network is trained on ONLY the 10 holdout
    # rows -- nothing else to fit, so it trivially memorizes them. This is
    # NOT a predictor: train set == test set == the same 10 known answers.
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: 'Cheating' NN (trained on ONLY the 10 holdout rows -- pure memorization)")
    print("=" * 60)
    cheat_hits = 0
    cheat_probs_by_digit = {
        col: train_predict(holdout_feats, holdout_labels, holdout_feats, col) for col in DIGIT_COLS
    }
    cheat_rows = []
    for i, row in enumerate(holdout_labels.itertuples()):
        probs = {col: cheat_probs_by_digit[col][i] for col in DIGIT_COLS}
        guesses = top_k_combos(probs, N_GUESSES)
        hit = row.combo in guesses
        cheat_hits += int(hit)
        cheat_rows.append({"date": row.date.date(), "actual": row.combo, "top_guess": guesses[0], "hit": hit})
    print(pd.DataFrame(cheat_rows).to_string(index=False))
    print(f"\n'Cheating' NN: {cheat_hits}/{HOLDOUT_N} exact matches "
          f"({cheat_hits/HOLDOUT_N*100:.0f}%) -- achieved by memorizing the answers it was "
          f"trained on, not by predicting anything. This number is meaningless as a forecast.")


if __name__ == "__main__":
    main()

"""
Manual grid search for a Random Forest under the TSTR protocol:
train on synthetic data, score against real data, for every combination
in HYPERPARAMETERS.

Real data is split into a VALIDATION chunk (used to pick the best
hyperparameters) and a held-out TEST chunk (touched only once, after
the search, to report final unbiased numbers). This avoids tuning
directly against the same real data you report results on.

Features are scaled with a MinMaxScaler that is fit ONLY on the
synthetic training data (the "training set" in the TSTR protocol) and
then applied unchanged to the real validation/test data. Fitting the
scaler on real data would leak real-data statistics into an experiment
whose whole point is testing generalization from synthetic-only
training.

Requires the same inputs as train_rf_tstr.py:
  - synthetic_samples.csv covering every class (see
    generate_synthetic_samples.py -> SAMPLES_PER_CLASS_ALL_CLASSES)
  - the original real CSV
  - wgan_gp_numerical_cols_protocol.pkl

Usage:
    python train_rf_tstr_gridsearch.py
"""

import itertools
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG - edit this section
# ============================================================

SYNTHETIC_CSV_PATH = "synthetic_samples.csv"
REAL_CSV_PATH = "ids2018-preprocessed-nonbenign.csv"
NUMERICAL_COLS_PKL = "wgan_gp_numerical_cols_protocol.pkl"

LABEL_COL = "Label"

# Real rows are sampled down (stratified) to this many before being split
# into validation/test, since the full CSV is ~1.3M rows. Set to None to
# use the full real dataset (slower).
REAL_SAMPLE_SIZE = 150_000

# Of that sample, this fraction goes to validation (used during the grid
# search); the rest becomes the held-out test set (used once, at the end).
REAL_VALIDATION_FRACTION = 0.5

RANDOM_STATE = 42

# Metric used to rank hyperparameter combinations on the validation set.
# One of: "f1_macro", "balanced_accuracy", "accuracy"
SELECTION_METRIC = "f1_macro"

HYPERPARAMETERS = {
    "n_estimators": [100, 150],
    "max_depth": [18, 22, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "ccp_alpha": [0.0, 0.0001],
}

# Fixed RF settings applied to every combination (not searched over)
FIXED_RF_PARAMS = dict(
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

GRID_RESULTS_CSV = "tstr_gridsearch_results.csv"
CONFUSION_MATRIX_OUTPUT_PNG = "tstr_gridsearch_best_confusion_matrix.png"
CLASSIFICATION_REPORT_OUTPUT_CSV = "tstr_gridsearch_best_classification_report.csv"
MODEL_OUTPUT_PKL = "rf_model_best.pkl"
SCALER_OUTPUT_PKL = "tstr_min_max_scaler.pkl"


# ============================================================

def build_feature_frame(df, numerical_cols):
    """Select numerical_cols + Protocol and coerce Protocol to numeric."""
    feature_cols = numerical_cols + ["Protocol"]
    X = df[feature_cols].copy()
    X["Protocol"] = pd.to_numeric(X["Protocol"], errors="coerce")
    return X


def score(y_true, y_pred, metric_name):
    if metric_name == "f1_macro":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    raise ValueError(f"Unknown SELECTION_METRIC: {metric_name}")


def main():
    with open(NUMERICAL_COLS_PKL, "rb") as f:
        numerical_cols = pickle.load(f)

    print("Loading synthetic data:", SYNTHETIC_CSV_PATH)
    synth_df = pd.read_csv(SYNTHETIC_CSV_PATH)

    print("Loading real data:", REAL_CSV_PATH)
    real_df = pd.read_csv(REAL_CSV_PATH)

    synth_classes = set(synth_df[LABEL_COL].unique())
    real_classes = set(real_df[LABEL_COL].unique())
    missing_from_synth = real_classes - synth_classes
    if missing_from_synth:
        print(
            "\nWARNING: these real classes have NO synthetic samples, so "
            "the RF will never be able to predict them:\n  "
            f"{sorted(missing_from_synth)}\n"
        )

    # ---- sample + split real data into validation / test ----
    if REAL_SAMPLE_SIZE is not None and len(real_df) > REAL_SAMPLE_SIZE:
        real_df, _ = train_test_split(
            real_df,
            train_size=REAL_SAMPLE_SIZE,
            stratify=real_df[LABEL_COL],
            random_state=RANDOM_STATE,
        )

    real_val_df, real_test_df = train_test_split(
        real_df,
        train_size=REAL_VALIDATION_FRACTION,
        stratify=real_df[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    X_train = build_feature_frame(synth_df, numerical_cols)
    y_train = synth_df[LABEL_COL]

    X_val = build_feature_frame(real_val_df, numerical_cols)
    y_val = real_val_df[LABEL_COL]

    X_test = build_feature_frame(real_test_df, numerical_cols)
    y_test = real_test_df[LABEL_COL]

    print(f"\nTrain (synthetic):        {len(X_train)} rows")
    print(f"Validation (real, search): {len(X_val)} rows")
    print(f"Test (real, held out):     {len(X_test)} rows")

    # ---- scale numerical features ----
    # Fit ONLY on the synthetic training data, then apply that same fitted
    # transform to the real validation/test data. "Protocol" is a
    # categorical code rather than a continuous quantity, so it's left
    # unscaled - same treatment the production pipeline gives its one-hot
    # protocol columns.
    scaler = MinMaxScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    with open(SCALER_OUTPUT_PKL, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler fit on synthetic training data saved to {SCALER_OUTPUT_PKL}")

    # ---- build the grid ----
    keys = list(HYPERPARAMETERS.keys())
    combinations = list(itertools.product(*HYPERPARAMETERS.values()))
    print(f"\nTotal hyperparameter combinations: {len(combinations)}")

    grid_results = []
    best_score = -np.inf
    best_params = None
    best_model = None

    for i, combo in enumerate(combinations, start=1):
        params = dict(zip(keys, combo))

        print(f"\n[{i}/{len(combinations)}] Training with: {params}")

        clf = RandomForestClassifier(**params, **FIXED_RF_PARAMS)
        clf.fit(X_train, y_train)

        y_val_pred = clf.predict(X_val)

        val_metrics = {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
            "f1_macro": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_val, y_val_pred, average="weighted", zero_division=0),
        }

        current_score = val_metrics[SELECTION_METRIC]

        print(f"  val {SELECTION_METRIC} = {current_score:.4f} "
              f"(acc={val_metrics['accuracy']:.4f}, "
              f"bal_acc={val_metrics['balanced_accuracy']:.4f})")

        grid_results.append({**params, **val_metrics})

        pd.DataFrame(grid_results).to_csv(GRID_RESULTS_CSV, index=False)

        if current_score > best_score:
            best_score = current_score
            best_params = params
            best_model = clf

    grid_results_df = pd.DataFrame(grid_results).sort_values(
        SELECTION_METRIC, ascending=False
    )
    grid_results_df.to_csv(GRID_RESULTS_CSV, index=False)

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best params (by val {SELECTION_METRIC} = {best_score:.4f}):")
    print(best_params)
    print(f"\nFull grid results saved to {GRID_RESULTS_CSV}")

    # ---- final, unbiased evaluation on held-out real test set ----
    y_test_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("FINAL TSTR RESULTS ON HELD-OUT REAL TEST SET")
    print("(best model from grid search, evaluated on data the search "
          "never saw)")
    print("=" * 60)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    report_dict = classification_report(
        y_test, y_test_pred, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_OUTPUT_CSV)
    print(f"\nFull per-class report saved to "
          f"{CLASSIFICATION_REPORT_OUTPUT_CSV}")
    print(report_df.round(4))

    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_test_pred, labels=labels_sorted)

    plt.figure(figsize=(max(6, len(labels_sorted) * 0.6),
                         max(5, len(labels_sorted) * 0.5)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels_sorted, yticklabels=labels_sorted
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("TSTR Confusion Matrix - Best Grid Search Model (held-out real test)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_OUTPUT_PNG}")

    # ---- save the best model, separate from the scaler saved above ----
    # Saved this way (two separate pickle files) so it matches exactly how
    # test_rf_unlabeled.py loads a model and scaler: two independent
    # pickle.load() calls.
    with open(MODEL_OUTPUT_PKL, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {MODEL_OUTPUT_PKL}")


if __name__ == "__main__":
    main()

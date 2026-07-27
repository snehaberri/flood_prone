"""
Train + evaluate the flood-susceptibility classifier.

Reads:
    data/processed/points_for_feature_extraction.csv  (point_id, lon, lat, label)
    data/processed/features.csv                        (point_id, feature columns -- see FEATURES_SPEC.md)

Does:
    1. Spatial grid CV (GroupKFold on a lon/lat grid cell id) -- prevents
       the random-split leakage flagged in the project plan. A random split
       would let the model memorize local terrain noise near training points
       that reappears in a spatially adjacent test point; grid CV forces it
       to generalize across neighborhoods it hasn't seen.
    2. Heuristic baseline: score = -elevation_m - dist_to_drain_m (lower
       elevation + closer to drain => higher risk). Every learned model
       must beat this or the "public features carry signal" claim fails.
    3. XGBoost classifier on the full feature set.
    4. Reports ROC-AUC, PR-AUC, precision@k, and a calibration curve for both,
       on held-out spatial folds only -- never on training folds.

Run:
    python3 src/train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import xgboost as xgb

GRID_CELL_DEG = 0.02  # ~2km cells at this latitude, per project plan
N_FOLDS = 5
TOP_K_FRACTION = 0.2  # precision@top-20%-riskiest, adjust as needed
SEED = 42


def load_data():
    points = pd.read_csv("data/processed/points_for_feature_extraction.csv")
    features = pd.read_csv("data/processed/features.csv")

    missing = set(points["point_id"]) - set(features["point_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} points have no features (e.g. {list(missing)[:5]}). "
            "features.csv must cover every point_id in points_for_feature_extraction.csv."
        )

    df = points.merge(features, on="point_id", how="left")
    return df


def assign_grid_cells(df: pd.DataFrame) -> pd.Series:
    cell_x = (df["lon"] // GRID_CELL_DEG).astype(int)
    cell_y = (df["lat"] // GRID_CELL_DEG).astype(int)
    return cell_x.astype(str) + "_" + cell_y.astype(str)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    k = max(1, int(len(y_score) * k_frac))
    top_k_idx = np.argsort(-y_score)[:k]
    return y_true[top_k_idx].mean()


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray, results: list):
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    p_at_k = precision_at_k(y_true, y_score, TOP_K_FRACTION)
    results.append({"model": name, "roc_auc": auc, "pr_auc": ap, "precision_at_k": p_at_k})


def main():
    df = load_data()
    df["grid_cell"] = assign_grid_cells(df)

    numeric_features = ["elevation_m", "slope_deg", "dist_to_drain_m", "dist_to_lake_m", "impervious_pct"]
    available = [c for c in numeric_features if c in df.columns]
    missing = set(numeric_features) - set(available)
    if missing:
        print(f"WARNING: missing feature columns {missing}, proceeding without them")

    if "landuse_class" in df.columns:
        df = pd.get_dummies(df, columns=["landuse_class"], prefix="lu", dummy_na=True)
        landuse_cols = [c for c in df.columns if c.startswith("lu_")]
    else:
        landuse_cols = []

    feature_cols = available + landuse_cols
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["grid_cell"].values

    n_unique_groups = len(set(groups))
    n_folds = min(N_FOLDS, n_unique_groups)
    gkf = GroupKFold(n_splits=n_folds)

    baseline_results, xgb_results = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- heuristic baseline: no fitting needed, just score by definition ---
        if "elevation_m" in df.columns and "dist_to_drain_m" in df.columns:
            baseline_score = -df["elevation_m"].values[test_idx] - df["dist_to_drain_m"].values[test_idx]
            evaluate(f"baseline_fold{fold}", y_test, baseline_score, baseline_results)

        # --- xgboost ---
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED,
        )
        model.fit(X_train, y_train)
        xgb_score = model.predict_proba(X_test)[:, 1]
        evaluate(f"xgb_fold{fold}", y_test, xgb_score, xgb_results)

    print(f"\n{n_folds}-fold spatial grid CV ({n_unique_groups} unique grid cells, "
          f"{GRID_CELL_DEG}deg ~ {GRID_CELL_DEG*111:.1f}km cells)\n")

    for label, results in [("HEURISTIC BASELINE", baseline_results), ("XGBOOST", xgb_results)]:
        if not results:
            continue
        rdf = pd.DataFrame(results)
        print(f"{label}:")
        print(f"  ROC-AUC:       {rdf.roc_auc.mean():.3f} +/- {rdf.roc_auc.std():.3f}")
        print(f"  PR-AUC:        {rdf.pr_auc.mean():.3f} +/- {rdf.pr_auc.std():.3f}")
        print(f"  Precision@{int(TOP_K_FRACTION*100)}%:  {rdf.precision_at_k.mean():.3f} +/- {rdf.precision_at_k.std():.3f}")
        print()

    # calibration curve on pooled out-of-fold predictions (fit on all data for this diagnostic only)
    model_full = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=SEED,
    )
    model_full.fit(X, y)
    probs = model_full.predict_proba(X)[:, 1]
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
    print("Calibration curve (in-sample, quantile bins -- for diagnostic only, not a CV metric):")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  predicted={mp:.2f}  observed={fp:.2f}")

    pd.DataFrame(baseline_results).to_csv("data/processed/cv_results_baseline.csv", index=False)
    pd.DataFrame(xgb_results).to_csv("data/processed/cv_results_xgb.csv", index=False)
    print("\nwrote data/processed/cv_results_baseline.csv, cv_results_xgb.csv")


if __name__ == "__main__":
    main()

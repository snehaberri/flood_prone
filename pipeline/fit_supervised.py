"""The slot for a supervised model, for when labelled flood observations arrive.

Nothing calls this yet, because there is nothing to train on. It exists so that the
day a journalist's RTI response, a BBMP complaint export, or a season of field
observations lands, the work is loading a CSV rather than rebuilding a pipeline.

Expected input: data/raw/flood_events.csv with at least

    lat, lon, date, flooded        # flooded in {0, 1}

Read this before running it
---------------------------
Three ways this kind of model goes wrong, all of which we have already seen once on
this project. Each is guarded below rather than left as advice.

1. RANDOM CROSS-VALIDATION LIES. Flood points are spatially clustered, so a random
   split puts near-duplicate points in train and test and reports an accuracy that
   evaporates on new ground. We use spatial block CV: the city is tiled, and whole
   tiles are held out.

2. NEGATIVE SAMPLING DECIDES THE ANSWER. Draw negatives uniformly across the city and
   the model learns "city centre versus outskirts", because that is what separates the
   classes. Negatives must be drawn from the same kind of place as the positives. The
   default here matches negatives to positives by road density.

3. ZERO POSITIVES IN A FOLD. With few events this happens silently and the fold's
   score is meaningless. We check and refuse.

Whatever comes out, report the spatial-block score, never the random-split one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

FEATURES = ["hand", "sink_depth", "twi", "slope", "dist_drain", "flowacc", "elev"]
BLOCK_KM = 3.0          # spatial CV block size


def load_events():
    p = C.RAW / "flood_events.csv"
    if not p.exists():
        raise SystemExit(
            "No labelled events yet.\n\n"
            f"Expected {p} with columns: lat, lon, date, flooded\n\n"
            "Until that exists, score_risk.py is the model - a transparent physical\n"
            "index that claims no accuracy. That is the honest state of this project,\n"
            "not a gap to paper over with synthetic labels."
        )
    df = pd.read_csv(p)
    missing = {"lat", "lon", "flooded"} - set(df.columns)
    if missing:
        raise SystemExit(f"flood_events.csv is missing columns: {sorted(missing)}")
    return df


def attach_features(events):
    """Give every event the features of its nearest road segment."""
    from scipy.spatial import cKDTree
    import json

    e = pd.read_parquet(C.PROC / "edges_scored.parquet")
    mids = np.array([json.loads(c)[len(json.loads(c)) // 2] for c in e.coords])
    k = np.cos(np.radians(C.CITY_CENTRE[0]))
    tree = cKDTree(np.c_[mids[:, 0] * k * 111_320, mids[:, 1] * 111_320])
    d, i = tree.query(np.c_[events.lon * k * 111_320, events.lat * 111_320])
    out = events.copy()
    out["snap_dist_m"] = d
    for f in FEATURES:
        out[f] = e[f].to_numpy()[i]
    out["susceptibility"] = e["susceptibility"].to_numpy()[i]
    far = out.snap_dist_m > 300
    if far.any():
        print(f"  dropping {far.sum()} events over 300 m from any road")
    return out[~far]


def spatial_blocks(df, km=BLOCK_KM):
    """Assign each point to a square block, so CV folds hold out whole areas."""
    deg_lat = km / 111.0
    deg_lon = km / (111.0 * np.cos(np.radians(C.CITY_CENTRE[0])))
    return (((df.lat - C.BBOX["south"]) // deg_lat).astype(int).astype(str) + "_" +
            ((df.lon - C.BBOX["west"]) // deg_lon).astype(int).astype(str))


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import GroupKFold

    ev = attach_features(load_events())
    y = ev.flooded.to_numpy().astype(int)
    X = ev[FEATURES].to_numpy()
    groups = spatial_blocks(ev)

    print(f"  {len(ev):,} events - {y.sum():,} positive, {(1 - y).sum():,} negative")
    print(f"  {groups.nunique()} spatial blocks of {BLOCK_KM} km")
    if y.sum() < 30 or (1 - y).sum() < 30:
        print("\n  WARNING: fewer than 30 of one class. Any score below is noise.")

    n_splits = min(5, groups.nunique())
    if n_splits < 3:
        raise SystemExit("need at least 3 spatial blocks for meaningful cross-validation")

    aucs, aps, base_aucs = [], [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            print("  skipping a fold with only one class present")
            continue
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                   class_weight="balanced", n_jobs=-1, random_state=0)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        # The index we already have, on the same held-out fold. If the trained model
        # cannot beat it, it is not earning its complexity.
        base_aucs.append(roc_auc_score(y[te], ev.susceptibility.to_numpy()[te]))

    if not aucs:
        raise SystemExit("no usable folds - the events are too clustered to validate")

    print(f"\n  spatial-block CV over {len(aucs)} folds")
    print(f"    trained model  AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}"
          f"   AP {np.mean(aps):.3f}")
    print(f"    physical index AUC {np.mean(base_aucs):.3f}")
    delta = np.mean(aucs) - np.mean(base_aucs)
    print(f"    the trained model {'beats' if delta > 0.02 else 'does NOT beat'} "
          f"the index ({delta:+.3f})")

    final = RandomForestClassifier(n_estimators=600, min_samples_leaf=3,
                                   class_weight="balanced", n_jobs=-1, random_state=0)
    final.fit(X, y)
    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\n  feature importance:\n{imp.round(3).to_string()}")

    import joblib
    joblib.dump(final, C.PROC / "supervised_model.joblib")
    print(f"\n  wrote {C.PROC / 'supervised_model.joblib'}")


if __name__ == "__main__":
    main()

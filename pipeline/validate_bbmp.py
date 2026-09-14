"""Check the susceptibility index against the BBMP point locations in this repo.

These are better than the geocoded locality centroids in `validate.py`: real named
points, city-wide, and 190 of them. But they split into two sets that must NOT be
pooled, which is the trap the original trial model fell into.

    flood_prone_locations   70 pts   BBMP's flood-prone list.   USABLE as a check.
    lowlying_locations     126 pts   A TERRAIN descriptor.      CIRCULAR - excluded.

Scoring a terrain-derived index against "places that are low-lying" partly measures
whether low ground is low. `train_model.py` pools all 190 as positives, which is a
third reason its reported metrics cannot be taken at face value, alongside the
synthetic features and the negative sampling.

Measured, the low-lying set actually scores *lower* than the flood-prone set
(AUC 0.63 vs 0.67), so pooling them dilutes rather than inflates. The objection is
still that the two sets mean different things and a model trained on their union is
not predicting what its author thinks it is - but the direction of the distortion
here is dilution, which is worth stating plainly rather than assuming.

Both sets are still undated. A location appears because it floods often, not because
of a recorded event, so this remains a sanity check and NOT a model accuracy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import config as C

RADIUS_M = 300.0        # tighter than validate.py: these are real points, not centroids


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def sample_points(df, tree, susc, radius=RADIUS_M):
    """Worst-road susceptibility within `radius` of each point."""
    k = np.cos(np.radians(C.CITY_CENTRE[0]))
    out = []
    for _, r in df.iterrows():
        q = [r.lon * k * 111_320, r.lat * 111_320]
        near = tree.query_ball_point(q, radius)
        if not near:
            continue
        s = susc[near]
        out.append({"name": r["name"], "n_roads": len(near),
                    "p90": float(np.percentile(s, 90)),
                    "median": float(np.median(s))})
    return pd.DataFrame(out)


def random_controls(n, rng, edges_lonlat):
    """Controls drawn from the road network itself.

    Sampling uniformly over the bounding box would put most controls on the rural
    fringe, and the index would separate them trivially - measuring "urban vs not",
    not flood risk. Drawing from road midpoints keeps controls on the same kind of
    ground as the positives.
    """
    idx = rng.choice(len(edges_lonlat), size=n, replace=False)
    return pd.DataFrame({"name": [f"control_{i}" for i in range(n)],
                         "lon": edges_lonlat[idx, 0], "lat": edges_lonlat[idx, 1]})


def main():
    pos_path = C.ROOT / "data" / "processed" / "positives.csv"
    if not pos_path.exists():
        raise SystemExit(f"missing {pos_path}")
    p = pd.read_csv(pos_path)

    flood = p[p.sources.str.contains("flood_prone_locations", na=False)].copy()
    lowly = p[p.sources.str.contains("lowlying_locations", na=False)].copy()
    print(f"  BBMP flood-prone points : {len(flood)}")
    print(f"  low-lying points        : {len(lowly)}  (circular - reported, not scored)")

    e = pd.read_parquet(C.PROC / "edges_scored.parquet")
    mids = np.array([json.loads(c)[len(json.loads(c)) // 2] for c in e.coords])
    k = np.cos(np.radians(C.CITY_CENTRE[0]))
    tree = cKDTree(np.c_[mids[:, 0] * k * 111_320, mids[:, 1] * 111_320])
    susc = e.susceptibility.to_numpy()

    rng = np.random.default_rng(0)
    ctrl = random_controls(400, rng, mids)

    f = sample_points(flood, tree, susc)
    c = sample_points(ctrl, tree, susc)
    l = sample_points(lowly, tree, susc)
    print(f"  matched to roads within {RADIUS_M:.0f} m: "
          f"{len(f)} flood-prone, {len(c)} controls, {len(l)} low-lying\n")

    results = {}
    for metric in ("p90", "median"):
        a = auc(f[metric], c[metric])
        results[f"auc_{metric}"] = a
        print(f"  {metric:<7} flood-prone median {f[metric].median():.3f} | "
              f"control median {c[metric].median():.3f} | AUC {a:.3f}")

    cut = float(np.quantile(susc, 0.8))
    hit = float((f.p90 >= cut).mean())
    chit = float((c.p90 >= cut).mean())
    print(f"\n  network 80th percentile = {cut:.3f}")
    print(f"    BBMP flood-prone above it : {hit:.0%}")
    print(f"    random controls above it  : {chit:.0%}")

    lauc = auc(l["p90"], c["p90"])
    print(f"\n  low-lying set, for completeness only: AUC {lauc:.3f}")
    print("    Not a check on the model - 'low ground scores as low ground' is partly")
    print("    circular. Shown so the effect of pooling it with the flood-prone set")
    print(f"    is visible: it scores {'below' if lauc < results['auc_p90'] else 'above'} "
          f"the flood-prone set, so pooling")
    print("    the two would dilute the measured signal, not inflate it.")

    print("\n  worst 8 BBMP flood-prone points (where the index disagrees most):")
    for _, r in f.nsmallest(8, "p90").iterrows():
        print(f"    {r['name'][:34]:<36} p90 {r.p90:.3f}  ({r.n_roads} roads)")

    out = {
        "n_flood_prone": int(len(f)), "n_controls": int(len(c)),
        "n_lowlying_excluded": int(len(l)),
        "radius_m": RADIUS_M,
        **{k2: float(v) for k2, v in results.items()},
        "flood_prone_in_top_quintile": hit, "control_in_top_quintile": chit,
        "lowlying_auc_circular_do_not_quote": lauc,
        "caveat": "Undated point locations. Sanity check, not a model accuracy. "
                  "Low-lying points excluded from scoring as circular.",
    }
    (C.PROC / "validation_bbmp.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    f.to_csv(C.PROC / "validation_bbmp.csv", index=False)
    print(f"\n  wrote validation_bbmp.json / .csv")


if __name__ == "__main__":
    main()

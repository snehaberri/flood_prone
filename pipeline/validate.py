"""Sanity-check the susceptibility index against independently documented locations.

Read the caveats in build_checkset.py before quoting anything from here. In short:
the check-set has no dates, is precise only to a locality centroid, is small, and is
geographically biased. It can tell us the index is not obviously wrong. It cannot
tell us the index is accurate, and the numbers below are NOT a model accuracy.

The weights in score_risk.py were fixed before this ran and are not adjusted in
response to it. If they were, this would stop being a check and become a fit.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# Roads within this radius of a locality centroid are taken to represent it.
RADIUS_M = 500.0


def auc(pos, neg):
    """Mann-Whitney U / rank AUC: P(a random flood-prone point scores above a control)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    n1, n2 = len(pos), len(neg)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n2))


def drainage_agreement(tol_m=60.0):
    """Independent check on the TERRAIN pipeline, needing no flood labels.

    The drainage network in drainage.tif is derived purely from the DEM by flow
    accumulation. OSM's mapped drains and streams are surveyed by people who have
    never seen our DEM. If the two agree far more than chance, the depression
    filling and D8 routing are doing something real.

    The chance baseline matters: a network covering 8% of the grid would match 8%
    of anything by luck. We compare observed agreement against that rate.
    """
    import rasterio
    from scipy import ndimage

    with rasterio.open(C.PROC / "drainage.tif") as s:
        drn = s.read(1) > 0
    with rasterio.open(C.PROC / "dist_drain.tif") as s:
        dist_osm = s.read(1)          # metres to the nearest OSM drain/water

    # Dilate the DEM-derived network by the tolerance, then ask what share of
    # OSM drain cells it covers.
    r = max(1, int(round(tol_m / C.PIXEL_M)))
    grown = ndimage.binary_dilation(drn, ndimage.generate_binary_structure(2, 2),
                                    iterations=r)
    osm_cells = dist_osm <= C.PIXEL_M          # cells on a mapped OSM drain
    if osm_cells.sum() == 0:
        return None

    hit = float((osm_cells & grown).sum() / osm_cells.sum())
    chance = float(grown.sum() / np.isfinite(dist_osm).sum())
    return {"tolerance_m": tol_m, "osm_drain_cells": int(osm_cells.sum()),
            "matched": hit, "chance": chance,
            "lift": hit / chance if chance else float("nan")}


def main():
    cs = json.loads((C.PROC / "checkset.json").read_text(encoding="utf-8"))
    pts = cs["points"]
    e = pd.read_parquet(C.PROC / "edges_scored.parquet")

    # Edge midpoints, in an equal-ish metric space for radius queries.
    mids = np.array([json.loads(c)[len(json.loads(c)) // 2] for c in e.coords])
    k = np.cos(np.radians(C.CITY_CENTRE[0]))
    tree = cKDTree(np.c_[mids[:, 0] * k * 111_320, mids[:, 1] * 111_320])

    rows = []
    for p in pts:
        q = [p["lon"] * k * 111_320, p["lat"] * 111_320]
        near = tree.query_ball_point(q, RADIUS_M)
        if not near:
            print(f"  ! no roads within {RADIUS_M:.0f} m of {p['name']}")
            continue
        s = e.susceptibility.to_numpy()[near]
        rows.append({
            "name": p["name"], "label": p["label"], "n_roads": len(near),
            # The worst road in the locality, not the average: flooding is reported
            # when one junction goes under, not when a neighbourhood mean rises.
            "score_p90": float(np.percentile(s, 90)),
            "score_median": float(np.median(s)),
        })

    df = pd.DataFrame(rows)
    pos = df[df.label == "flood_prone"]
    neg = df[df.label == "control"]

    print(f"\n  check-set: {len(pos)} documented flood-prone, {len(neg)} control")
    print(f"  radius {RADIUS_M:.0f} m, median {df.n_roads.median():.0f} roads per point\n")

    for metric in ("score_p90", "score_median"):
        a = auc(pos[metric], neg[metric])
        print(f"  {metric}:")
        print(f"    flood-prone median {pos[metric].median():.3f} | "
              f"control median {neg[metric].median():.3f}")
        print(f"    rank AUC {a:.3f}   "
              f"({'above' if a > 0.5 else 'at or below'} chance)")

    # How many documented locations land in the riskiest fifth of the road network?
    cut = e.susceptibility.quantile(0.8)
    hit = (pos.score_p90 >= cut).mean()
    ctrl_hit = (neg.score_p90 >= cut).mean() if len(neg) else float("nan")
    print(f"\n  network-wide 80th percentile = {cut:.3f}")
    print(f"    flood-prone localities at or above it : {hit:.0%}")
    print(f"    control localities at or above it     : {ctrl_hit:.0%}")

    print("\n  ranked check-set:")
    for _, r in df.sort_values("score_p90", ascending=False).iterrows():
        flag = "F" if r.label == "flood_prone" else "."
        print(f"    {flag}  {r['name']:<22} p90 {r.score_p90:.3f}  ({r.n_roads} roads)")

    print("\n  terrain cross-check (no flood labels involved):")
    da = drainage_agreement()
    if da:
        print(f"    DEM-derived drainage vs {da['osm_drain_cells']:,} OSM-mapped drain cells")
        print(f"    matched within {da['tolerance_m']:.0f} m : {da['matched']:.1%}")
        print(f"    expected by chance             : {da['chance']:.1%}")
        print(f"    lift                           : {da['lift']:.2f}x")

    df.to_csv(C.PROC / "validation.csv", index=False)
    summary = {
        "n_flood_prone": int(len(pos)), "n_control": int(len(neg)),
        "radius_m": RADIUS_M,
        "drainage_agreement": da,
        "auc_p90": auc(pos.score_p90, neg.score_p90),
        "auc_median": auc(pos.score_median, neg.score_median),
        "flood_prone_in_top_quintile": float(hit),
        "control_in_top_quintile": float(ctrl_hit),
        "caveat": "Sanity check on a small, undated, centroid-precision check-set. "
                  "NOT a model accuracy. Weights were fixed before this ran.",
    }
    (C.PROC / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote validation.csv and validation.json")


if __name__ == "__main__":
    main()

"""
Sample negative (non-flood-prone) locations for the susceptibility classifier.

Strategy: uniform-random points across the Bengaluru urban extent, rejecting
any candidate within MIN_DIST_M of a known positive. This is deliberately
simple and auditable -- the risk isn't the sampling mechanic, it's an
unjustified choice of MIN_DIST_M (too small -> label noise from unlisted
flood-prone points near the boundary; too large -> negatives are trivially
separable from positives on elevation alone, inflating apparent model skill).

We therefore don't hardcode one threshold. We generate negative sets at
several thresholds and leave the choice + justification to the modeling
notebook, where it can be tied to actual CV performance rather than picked
a priori. This IS the "sensitivity check" the project plan calls for --
it's not a separate follow-up step.

NOTE ON EXTENT: BBMP administrative ward boundaries are not in this
container (no network access to opencity.in from the sandbox). Using a
padded bounding box over the known positives as a stand-in extent. This
is a real limitation: it doesn't guarantee negatives fall on actual city
land (a sample could technically land in a reservoir or outside BBMP
limits). Once you have the BBMP ward boundary shapefile, swap
`sample_within_bbox` for a proper `gpd.sjoin` against the ward polygon --
the interface below is written so that's a one-line change.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# padded bbox around known positives (see note above re: real ward boundary)
BBOX_PAD_DEG = 0.05  # ~5km pad
NEG_TO_POS_RATIO = 3  # generate 3x negatives per positive, before dedup/filtering
MIN_DIST_THRESHOLDS_M = [200, 300, 500]  # sensitivity sweep, per project plan
UTM_CRS = "EPSG:32643"
SEED = 42


def load_positives():
    gdf = gpd.read_file("data/processed/positives.geojson")
    return gdf.to_crs(UTM_CRS)


def get_bbox(positives_utm: gpd.GeoDataFrame, pad_deg: float):
    positives_wgs = positives_utm.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = positives_wgs.total_bounds
    return minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg


def sample_within_bbox(bbox_wgs, n: int, rng: np.random.Generator) -> gpd.GeoDataFrame:
    """Placeholder extent sampler. Replace with sjoin against BBMP ward
    polygon when that shapefile is available -- see module docstring."""
    minx, miny, maxx, maxy = bbox_wgs
    lons = rng.uniform(minx, maxx, n)
    lats = rng.uniform(miny, maxy, n)
    pts = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )
    return pts.to_crs(UTM_CRS)


def filter_by_min_distance(
    candidates_utm: gpd.GeoDataFrame,
    positives_utm: gpd.GeoDataFrame,
    min_dist_m: float,
) -> gpd.GeoDataFrame:
    """Keep candidates whose nearest positive is >= min_dist_m away.
    Uses sjoin_nearest (one row per candidate, no fan-out) rather than a
    buffer-based sjoin, which can duplicate rows when a candidate falls
    inside multiple positives' buffers."""
    nearest = gpd.sjoin_nearest(
        candidates_utm, positives_utm[["geometry"]], distance_col="dist_to_positive"
    )
    nearest = nearest[~nearest.index.duplicated(keep="first")]
    keep = nearest[nearest["dist_to_positive"] >= min_dist_m]
    return candidates_utm.loc[keep.index]


def main():
    positives_utm = load_positives()
    bbox_wgs = get_bbox(positives_utm, BBOX_PAD_DEG)
    print(f"positives: {len(positives_utm)}")
    print(f"sampling bbox (lon/lat): {bbox_wgs}")

    rng = np.random.default_rng(SEED)
    n_target_per_positive = NEG_TO_POS_RATIO
    n_positives = len(positives_utm)

    for min_dist in MIN_DIST_THRESHOLDS_M:
        # oversample generously since filtering rejects a chunk of candidates
        raw = sample_within_bbox(bbox_wgs, n_positives * n_target_per_positive * 3, rng)
        filtered = filter_by_min_distance(raw, positives_utm, min_dist)

        # also enforce negatives aren't within min_dist of EACH OTHER,
        # so we don't get clustered/pseudo-duplicate negatives
        filtered = filtered.sample(frac=1, random_state=SEED).reset_index(drop=True)
        kept_idx = []
        kept_geoms = []
        for i, geom in enumerate(filtered.geometry):
            if all(geom.distance(g) >= min_dist for g in kept_geoms):
                kept_idx.append(i)
                kept_geoms.append(geom)
            if len(kept_idx) >= n_positives * n_target_per_positive:
                break

        result = filtered.iloc[kept_idx].copy()
        result_wgs = result.to_crs("EPSG:4326")
        result_wgs["lon"] = result_wgs.geometry.x
        result_wgs["lat"] = result_wgs.geometry.y
        result_wgs["label"] = 0
        result_wgs["min_dist_threshold_m"] = min_dist

        out_path = f"data/processed/negatives_mindist{min_dist}m.csv"
        result_wgs.drop(columns="geometry").to_csv(out_path, index=False)
        print(f"  min_dist={min_dist}m: kept {len(result_wgs)} negatives -> {out_path}")


if __name__ == "__main__":
    main()

"""Turn raw OSM ways into a routable graph whose edges carry flood features.

Steps:
  1. Parse the road ways and split them at shared nodes, so the result is a real
     topological graph rather than a pile of overlapping polylines.
  2. Rasterise the OSM waterway and water-body layers, and compute a Euclidean
     distance-to-drainage raster from them.
  3. For every edge, sample the hydrology rasters along its length and reduce each
     to a statistic. We take the *worst* cell for the ponding indicators, not the
     mean: a 300 m road with one flooded underpass is impassable, and averaging
     hides exactly the thing we are trying to find.

Outputs:
    data/processed/edges.parquet   one row per road segment, with features
    data/processed/nodes.parquet   node id -> lon/lat
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
import config as C

TO_M = Transformer.from_crs(C.CRS_WGS, C.CRS_M, always_xy=True).transform

# Sample the rasters this often along each edge, in metres.
SAMPLE_STEP_M = 20.0

# Nominal free-flow speeds by OSM highway class, km/h. Used for travel-time routing.
SPEED_KMH = {
    "motorway": 80, "motorway_link": 50, "trunk": 60, "trunk_link": 40,
    "primary": 50, "primary_link": 35, "secondary": 40, "secondary_link": 30,
    "tertiary": 35, "tertiary_link": 25, "unclassified": 30, "residential": 25,
}
DEFAULT_SPEED = 25


def load_ways(name):
    p = C.RAW / f"osm_{name}.json"
    if not p.exists():
        return []
    els = json.loads(p.read_text(encoding="utf-8"))["elements"]
    return [e for e in els if e.get("type") == "way" and e.get("geometry")]


def split_ways_at_junctions(ways):
    """Split each way where it meets another way, producing graph edges."""
    shared = Counter()
    for w in ways:
        for nd in w.get("nodes", []):
            shared[nd] += 1

    edges, nodes = [], {}
    for w in ways:
        nds = w.get("nodes") or []
        geom = w["geometry"]
        if len(nds) != len(geom) or len(nds) < 2:
            continue
        for nd, g in zip(nds, geom):
            nodes[nd] = (g["lon"], g["lat"])

        tags = w.get("tags", {})
        oneway = tags.get("oneway") in ("yes", "true", "1") or tags.get("junction") == "roundabout"
        hw = tags.get("highway", "residential")

        # Break points: the ends, plus every interior node shared with another way.
        cuts = [0] + [k for k in range(1, len(nds) - 1) if shared[nds[k]] > 1] + [len(nds) - 1]
        for a, b in zip(cuts, cuts[1:]):
            if b <= a:
                continue
            edges.append({
                "u": nds[a], "v": nds[b],
                "way_id": w["id"], "highway": hw,
                "name": tags.get("name", ""),
                "oneway": bool(oneway),
                "coords": [(g["lon"], g["lat"]) for g in geom[a:b + 1]],
            })
    return edges, nodes


def drainage_distance(profile):
    """Distance in metres from every cell to the nearest mapped drain or water body."""
    shapes = []
    for layer in ("waterways", "water"):
        for w in load_ways(layer):
            pts = [(g["lon"], g["lat"]) for g in w["geometry"]]
            if len(pts) < 2:
                continue
            line = shp_transform(TO_M, LineString(pts))
            shapes.append(line)
    print(f"  drainage features rasterised: {len(shapes)}")
    if not shapes:
        return np.full((profile["height"], profile["width"]), np.nan, "float32"), 0

    mask = rasterize(
        ((g, 1) for g in shapes),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"], fill=0, dtype="uint8", all_touched=True,
    )
    dist = ndimage.distance_transform_edt(mask == 0, sampling=C.PIXEL_M).astype("float32")
    return dist, int(mask.sum())


def sample_along(edges_m, rasters, profile):
    """Sample every raster along every edge in one vectorised pass.

    All sample points from all edges go into a single array, get converted to row/col
    once, and are read with fancy indexing. Per-edge windowed reads would be ~100k
    round trips through rasterio.
    """
    inv = ~profile["transform"]
    xs, ys, owner = [], [], []
    for i, geom in enumerate(edges_m):
        n = max(2, int(geom.length // SAMPLE_STEP_M) + 1)
        for d in np.linspace(0, geom.length, n):
            p = geom.interpolate(d)
            xs.append(p.x)
            ys.append(p.y)
            owner.append(i)

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    owner = np.asarray(owner)
    cols, rows = inv * (xs, ys)
    rows = np.clip(rows.astype(int), 0, profile["height"] - 1)
    cols = np.clip(cols.astype(int), 0, profile["width"] - 1)
    print(f"  {len(owner):,} sample points across {len(edges_m):,} edges")

    n_edges = len(edges_m)
    out = {}
    for name, (arr, how) in rasters.items():
        vals = arr[rows, cols].astype("float64")
        vals = np.where(np.isfinite(vals), vals, np.nan)
        # Group by edge. np.maximum.at / add.at handle the scatter-reduce.
        if how == "max":
            acc = np.full(n_edges, -np.inf)
            np.maximum.at(acc, owner, np.nan_to_num(vals, nan=-np.inf))
            acc[~np.isfinite(acc)] = np.nan
        elif how == "min":
            acc = np.full(n_edges, np.inf)
            np.minimum.at(acc, owner, np.nan_to_num(vals, nan=np.inf))
            acc[~np.isfinite(acc)] = np.nan
        else:                                   # mean
            s = np.zeros(n_edges)
            c = np.zeros(n_edges)
            good = np.isfinite(vals)
            np.add.at(s, owner[good], vals[good])
            np.add.at(c, owner[good], 1.0)
            acc = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        out[name] = acc
    return out


def main():
    ways = load_ways("roads")
    if not ways:
        raise SystemExit("no road ways found - run fetch_osm.py first")
    print(f"  {len(ways):,} road ways")

    edges, nodes = split_ways_at_junctions(ways)
    print(f"  split into {len(edges):,} edges, {len(nodes):,} nodes")

    with rasterio.open(C.PROC / "dem.tif") as s:
        profile = s.profile

    print("  building distance-to-drainage raster ...", flush=True)
    dist_drain, n_cells = drainage_distance(profile)
    with rasterio.open(C.PROC / "dist_drain.tif", "w",
                       **dict(profile, dtype="float32", count=1,
                              compress="deflate", nodata=np.nan)) as d:
        d.write(dist_drain, 1)
    print(f"  drainage occupies {n_cells:,} cells; "
          f"median distance {np.median(dist_drain):.0f} m")

    rasters = {}
    for key, fname, how in [
        ("sink_depth", "sink_depth.tif", "max"),     # worst pond on the segment
        ("hand", "hand.tif", "min"),                 # closest approach to drainage level
        ("twi", "twi.tif", "max"),
        ("slope", "slope.tif", "mean"),
        ("flowacc", "flowacc.tif", "max"),
        ("elev", "dem_smooth.tif", "mean"),
    ]:
        with rasterio.open(C.PROC / fname) as s:
            rasters[key] = (s.read(1), how)
    rasters["dist_drain"] = (dist_drain, "min")

    geoms_m = [shp_transform(TO_M, LineString(e["coords"])) for e in edges]
    lengths = np.array([g.length for g in geoms_m])

    print("  sampling rasters along edges ...", flush=True)
    feats = sample_along(geoms_m, rasters, profile)

    df = pd.DataFrame({
        "u": [e["u"] for e in edges],
        "v": [e["v"] for e in edges],
        "highway": [e["highway"] for e in edges],
        "name": [e["name"] for e in edges],
        "oneway": [e["oneway"] for e in edges],
        "length_m": lengths,
        "coords": [json.dumps([[round(x, 6), round(y, 6)] for x, y in e["coords"]])
                   for e in edges],
        **feats,
    })
    df["speed_kmh"] = df["highway"].map(SPEED_KMH).fillna(DEFAULT_SPEED)
    df["base_time_s"] = df["length_m"] / (df["speed_kmh"] * 1000 / 3600)

    # Drop degenerate edges: zero length, or no terrain data (outside the DEM).
    before = len(df)
    df = df[(df.length_m > 1) & df.sink_depth.notna() & df.hand.notna()].reset_index(drop=True)
    print(f"  dropped {before - len(df):,} degenerate/off-DEM edges")

    df.to_parquet(C.PROC / "edges.parquet", index=False)
    nd = pd.DataFrame([{"node": k, "lon": v[0], "lat": v[1]} for k, v in nodes.items()])
    nd.to_parquet(C.PROC / "nodes.parquet", index=False)

    print(f"\n  edges {len(df):,}  nodes {len(nd):,}")
    print(df[["length_m", "sink_depth", "hand", "twi", "slope", "dist_drain"]]
          .describe().round(2).to_string())


if __name__ == "__main__":
    main()

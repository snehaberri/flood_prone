"""
Parse BBMP flood-prone / low-lying KML placemark lists into a single
deduplicated positives layer for the susceptibility model.

Input:
    data/raw/flood_prone_locations.kml   (OBJECTID, name, Point)
    data/raw/lowlying_locations.kml      (OBJECTID, name, Point)

Output:
    data/processed/positives.geojson
    data/processed/positives.csv

Dedup logic: the two BBMP lists are known to overlap (same underlying
"flood risk exercise", published as two separate resources on OpenCity).
We merge them and collapse points within DEDUP_RADIUS_M of each other
into a single positive, keeping a record of which source list(s)
contributed and how many raw points were collapsed (this count is itself
a signal -- locations flagged by both lists, or by the same list more
than once under slightly different names, are presumably higher-confidence
positives and are worth carrying through as a feature/weight later).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from lxml import etree

DEDUP_RADIUS_M = 150  # points closer than this are treated as one location
UTM_CRS = "EPSG:32643"  # UTM zone 43N, correct for Bengaluru -- meters, not degrees


def parse_kml_placemarks(path: str, source_label: str) -> gpd.GeoDataFrame:
    """Minimal KML point parser -- avoids fiona's KML driver quirks/version issues."""
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = etree.parse(path)
    records = []
    dropped = 0
    for pm in tree.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        coord_el = pm.find(".//kml:Point/kml:coordinates", ns)
        if coord_el is None or coord_el.text is None:
            dropped += 1
            continue
        try:
            lon, lat, *_ = [float(x) for x in coord_el.text.strip().split(",")]
        except ValueError:
            dropped += 1
            continue
        if not (np.isfinite(lon) and np.isfinite(lat)):
            dropped += 1
            continue
        records.append({
            "name": (name_el.text or "").strip() if name_el is not None else "",
            "source": source_label,
            "geometry": Point(lon, lat),
        })
    if dropped:
        print(f"  [{source_label}] dropped {dropped} placemark(s) with missing/invalid coordinates "
              f"(source data defect, not a parsing issue)")
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def dedup_points(gdf: gpd.GeoDataFrame, radius_m: float) -> gpd.GeoDataFrame:
    """Union-find style spatial dedup: collapse mutually-close points into clusters,
    keep cluster centroid, track contributing sources/names/raw count."""
    gdf_m = gdf.to_crs(UTM_CRS).reset_index(drop=True)
    n = len(gdf_m)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # spatial join: points within radius_m of each other
    buffered = gdf_m.copy()
    buffered["geometry"] = buffered.geometry.buffer(radius_m)
    joined = gpd.sjoin(gdf_m, buffered, how="inner", predicate="within")
    for i, j in zip(joined.index, joined["index_right"]):
        union(i, j)

    gdf_m["cluster"] = [find(i) for i in range(n)]

    rows = []
    for cluster_id, group in gdf_m.groupby("cluster"):
        centroid = group.geometry.union_all().centroid
        rows.append({
            "name": group["name"].iloc[0],
            "all_names": "; ".join(sorted(set(group["name"]))),
            "sources": "; ".join(sorted(set(group["source"]))),
            "n_raw_points": len(group),
            "geometry": centroid,
        })
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=UTM_CRS)
    return out.to_crs("EPSG:4326")


def main():
    flood = parse_kml_placemarks(
        "data/raw/flood_prone_locations.kml", "flood_prone_locations"
    )
    lowlying = parse_kml_placemarks(
        "data/raw/lowlying_locations.kml", "lowlying_locations"
    )

    print(f"flood_prone_locations: {len(flood)} raw points")
    print(f"lowlying_locations:    {len(lowlying)} raw points")

    combined = pd.concat([flood, lowlying], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    deduped = dedup_points(combined, DEDUP_RADIUS_M)
    print(f"after {DEDUP_RADIUS_M}m dedup: {len(deduped)} unique positives")

    both_lists = deduped[deduped["sources"].str.contains(";")]
    print(f"  of which flagged by BOTH source lists: {len(both_lists)}")

    deduped["lon"] = deduped.geometry.x
    deduped["lat"] = deduped.geometry.y
    deduped["label"] = 1

    deduped.to_file("data/processed/positives.geojson", driver="GeoJSON")
    deduped.drop(columns="geometry").to_csv("data/processed/positives.csv", index=False)
    print("\nwrote data/processed/positives.geojson and positives.csv")


if __name__ == "__main__":
    main()

# Feature extraction handoff spec

`src/train_model.py` expects a file at `data/processed/features.csv` with
**exactly one row per `point_id`** from `data/processed/points_for_feature_extraction.csv`,
joined on `point_id`. Column contract:

| column                  | type  | units          | source                                                              |
|--------------------------|-------|----------------|----------------------------------------------------------------------|
| `point_id`                | str   | -              | must match points_for_feature_extraction.csv exactly, no drops       |
| `elevation_m`              | float | meters         | SRTM 30m DEM, sampled at point                                       |
| `slope_deg`                | float | degrees        | computed from DEM (e.g. `richdem.TerrainAttribute(dem, attrib='slope_riserun')`, then convert to degrees) |
| `dist_to_drain_m`           | float | meters         | nearest distance to BBMP stormwater drain line (primary/secondary/tertiary combined), or OSM `waterway=drain/ditch` if BBMP layer unavailable |
| `dist_to_lake_m`            | float | meters         | nearest distance to OSM `natural=water` polygon boundary, or Jal Dharohar water body layer |
| `impervious_pct`            | float | 0-100          | ESA WorldCover 10m, % impervious-surface pixels in a fixed buffer (suggest 100m radius) around point |
| `landuse_class`             | str   | categorical    | dominant OSM `landuse`/`natural` tag in same buffer (e.g. residential/commercial/industrial/park/water) |

Notes on doing this **without** network access from a sandbox:
- SRTM 30m tiles: download via the `elevation` Python package (`eio clip`) or directly from USGS EarthExplorer / OpenTopography, run locally where you have internet.
- OSM data: `osmnx.features_from_place("Bengaluru, India", tags={...})`, or a pre-clipped `.osm.pbf` from Geofabrik (south-asia extract) + `pyrosm`.
- BBMP stormwater drain shapefile: same OpenCity dataset page as the flood-prone KMLs — worth using instead of OSM drains since it's the authoritative source and matches what the flood-prone labels are actually adjacent to.
- ESA WorldCover: download the relevant 3°×3° tile covering Bengaluru (`ESA_WorldCover_10m_2021_v200_N12E075` or similar) from the WorldCover S3/AWS bucket or Copernicus browser.

Once you have `elevation_m`, `slope_deg`, `dist_to_drain_m`, `dist_to_lake_m`,
`impervious_pct`, `landuse_class` for all 764 points, drop the CSV at
`data/processed/features.csv` and hand it back — `train_model.py` does the rest
(join, spatial CV, baseline vs XGBoost, metrics, calibration).

If any feature turns out infeasible to compute locally (e.g. ESA WorldCover
access is a pain), tell me which one and I'll adjust the model to drop it
rather than block the whole pipeline on one layer.

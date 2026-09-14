# flood_prone — Bengaluru flood-risk mapping and routing

A flood susceptibility map for Bengaluru's road network, and a router that finds a way
around standing water. Everything runs from open data; no API key, account, or licence
negotiation is required.

```bash
pip install numpy pandas scipy rasterio geopandas shapely pyproj scikit-learn pyarrow
python run_all.py          # full pipeline, ~30-60 min on a cold start
python serve.py            # then open http://127.0.0.1:8000
```

The pipeline caches every download, so re-runs are cheap and an interrupted run resumes
where it stopped.

---

## Read this first

**This is not a trained flood classifier and it reports no accuracy.**

A supervised model needs labels: places where water actually stood, *with dates*. No such
public dataset exists for Bengaluru. What exists — including the BBMP point lists in
`data/raw/` — is static: a location is on the list because it floods often, not because
of a recorded event. Our RTIs and data requests have so far produced one reply, pointing
at a building-certification portal with no event data in it.

Given that, the susceptibility score is a **weighted-overlay index** over physically
meaningful terrain variables, with weights fixed *a priori* from the urban-pluvial
literature. Every term has a physical meaning, the weights live in one dictionary at the
top of `pipeline/score_risk.py`, and no fitted accuracy is claimed.

`pipeline/fit_supervised.py` is a working slot for real labels when they arrive. It
already enforces spatial-block cross-validation and density-matched negatives — the two
traps described below.

---

## What drives the score

| feature | weight | why it matters | source |
|---|---|---|---|
| Height above nearest drainage (HAND) | 0.25 | low ground near a drain floods first | DEM |
| Depression depth | 0.22 | closed sinks are where water physically sits | DEM |
| Topographic wetness index | 0.18 | standard saturation proxy | DEM |
| Slope | 0.15 | flat ground drains slowly | DEM |
| Distance to drain or lake | 0.12 | backflow and tank overflow | OSM |
| Flow accumulation | 0.08 | where upslope water converges | DEM |

Rainfall enters multiplicatively, so the map is time-varying:

```
risk(edge, t) = susceptibility(edge) × trigger(t)
cost(edge, t) = free_flow_time(edge) × (1 + 8 × risk(edge, t))
```

`trigger` runs 0 (dry) to 1 (saturating downpour), from Open-Meteo hourly rainfall
against IMD-style thresholds. With no rain the flood-aware route collapses onto the
fastest route. That is correct behaviour, not a bug.

Measured routing behaviour, Silk Board → Hebbal:

| rain trigger | roads blocked | fastest | safest | mean risk |
|---|---|---|---|---|
| 0.0 (dry) | 0 | 15.98 km | 15.98 km — identical | 0.000 |
| 0.3 | 0 | 15.98 km | 16.35 km | 0.167 → 0.158 |
| 0.6 | 3,140 | 15.98 km | 20.78 km | 0.335 → 0.230 |
| 0.9 | 33,081 | 15.98 km | 16.22 km | 0.502 → 0.444 |

At trigger 0.9 the detour is *shorter* than at 0.6 because so many roads are blocked that
the long way round is itself cut; the router falls back to the least-bad path and flags it.

---

## How good is it, honestly

Three checks. **None of them is an accuracy figure.**

**1. Terrain cross-check — the most credible number here.** The drainage network derived
purely from the DEM by flow accumulation, compared against OSM's hand-surveyed drains.
**72.3% of OSM drainage cells fall within 60 m of a DEM-derived drainage line, against a
32.9% chance baseline — 2.20× lift.** No labels, no judgement weights, two fully
independent sources. Against drains and streams only (excluding areal lake polygons, which
flow lines do not cross) it is 76.4% at 2.32×.

**2. Point check-sets — modest signal.** Two independent sets agree:

| check-set | n | AUC (p90) | AUC (median) |
|---|---|---|---|
| BBMP flood-prone points (`validate_bbmp.py`) | 70 vs 400 controls | 0.669 | 0.604 |
| Localities in 2022/24 flood reporting (`validate.py`) | 30 vs 10 controls | 0.673 | 0.753 |

**Above chance, and not by much.** 81% of BBMP flood-prone points sit above the network's
80th percentile — but so do 66% of random road-network controls. Silk Board Junction ranks
1st of 40 on the locality set and Bellandur 2nd, which is what you would hope. Against
that, Hebbal, Bommanahalli, Marathahalli and Rajajinagar all score below 0.51 despite
being well documented.

The misses are informative rather than random: Hebbal floods at the flyover underpass,
Bommanahalli and Marathahalli on the Outer Ring Road. Those are **drainage-capacity
failures at sub-30 m scale** — precisely what a topographic index on a 30 m surface model
cannot see. This is the structural ceiling on the current approach, and more weight-tuning
will not lift it.

**3. Weight sensitivity.** Every weight perturbed by ±30% over 200 draws: mean rank
correlation with baseline **0.996** (worst 0.985), and **94.4%** of the riskiest 5% of
roads retained. The a priori weights are not what is driving the result, which is the main
worry with a judgement-set index.

---

## Issues found in the original trial model (`src/`)

`src/` is the first-pass model and is **superseded**. Its reported metrics should not be
quoted. Three independent problems, each sufficient on its own:

**1. `data/processed/features.csv` is synthetic.** Nearest-neighbour feature differences
equal random-pair differences (ratio ≈ 1.00 across `elevation_m`, `slope_deg`,
`dist_to_drain_m`, `dist_to_lake_m`, `impervious_pct`). Real geography has spatial
autocorrelation; these values have none, so they were generated rather than sampled.
`cv_results_baseline.csv` and `cv_results_xgb.csv` inherit this and mean nothing.
`FEATURES_SPEC.md` itself is sound — it is the fulfilment that is fake.

**2. Negative sampling turned it into a "city centre vs outskirts" classifier.** Negatives
drawn by distance from positives are systematically further out, so the model learns
urban-versus-peripheral, which is what separates the classes. Negatives must come from the
same kind of place as positives — `fit_supervised.py` matches on road density.

**3. Positives pool two different things.** `positives.csv` merges 70
`flood_prone_locations` with 126 `lowlying_locations`. The second is a *terrain*
descriptor, so a terrain-derived model scored against it is partly circular. Measured, the
low-lying set scores *lower* (AUC 0.63 vs 0.67), so pooling dilutes rather than inflates —
but a model trained on the union is not predicting what it appears to.

A fourth, applying to any future model: **random cross-validation will lie.** Flood points
are spatially clustered, so a random split puts near-duplicates in train and test.
`fit_supervised.py` uses spatial block CV and refuses folds containing a single class.

## Bugs found and fixed while building the new pipeline

Recorded because each was silent — the code ran and produced plausible-looking output.

**D8 flow routing compared the wrong neighbour.** The array-shift slicing in
`d8_receivers()` read elevation from the neighbour *opposite* the one it recorded as the
receiver. Peak flow accumulation was **15 cells** on a 3.5 M-cell grid; correct is
**748,446** (21% of the domain). Every downstream product — TWI, HAND, the entire index —
was meaningless while looking entirely reasonable. `build_hydrology.py` now hard-fails if
peak accumulation falls below 1% of valid cells.

**Overpass reports failure as success.** A query it cannot finish in time may return
`200 OK` with zero elements rather than an error. Caching that silently deleted **1,203
water bodies — 58% of the total**, concentrated in the north and east and including Hebbal
Lake. `fetch_osm.py` now retries empty responses across every mirror before accepting one.
Keep that guard if you add a layer. (Restoring the lakes moved Hebbal's score by +0.006,
so this was not the cause of that miss.)

**DEM nodata filled as zero.** A lat/lon rectangle is not a rectangle in UTM, so the
destination grid corners fall outside the source tiles. Copernicus COGs declare
`nodata=None`, so those cells became `0` — read by the hydrology stage as 900 m-deep pits
that would have dominated the risk surface. Now initialised to NaN with a plausibility
guard.

**Route geometry was zigzagged.** Edges are stored `u→v` but Dijkstra walks them either
way; segments appended without reorientation produced discontinuous lines. Routes now
carry a `joint_gap_m` invariant, which must be ~0.

---

## Data sources

| what | source | licence | notes |
|---|---|---|---|
| Terrain | Copernicus DEM GLO-30 | free, open | AWS public bucket, no auth |
| Roads, drains, lakes | OpenStreetMap (Overpass) | ODbL | 147,530 road ways, 2,891 waterways, 2,041 water bodies |
| Rainfall | Open-Meteo | free, no key | ~11 km model, 16-point city grid |
| Flood-prone points | BBMP lists (`data/raw/*.kml`) | — | undated; sanity check only |

**The DEM is the weak link and the highest-value upgrade.** Copernicus GLO-30 is a
*surface* model — it includes rooftops and canopy. `build_hydrology.py` strips
building-scale spikes with a 60 m morphological opening, which is an approximation and the
largest single source of error in the terrain features. **FABDEM** is bare-earth and
purpose-built for this; it needs a licence accepted by hand so it cannot sit in an
automated pipeline, but swapping it in is a one-file change.

## Known limitations

- **No storm-drain capacity data.** Urban flooding is as much about blocked and undersized
  drains as about terrain. This is the missing axis, and it is what the outstanding data
  requests are for — not a nice-to-have.
- **Lake and tank levels are static.** Bengaluru's cascading tank network matters
  enormously — an upstream tank breaching changes everything downstream — and is not modelled.
- **No validation against actual flood events**, because no dated event data exists.
- **OSM drain coverage is partial**, so distance-to-drainage is optimistic where the
  rajakaluve network is unmapped.
- Rainfall is a ~11 km forecast model; Bengaluru convective cells are smaller than that.
- Roads are a snapshot — no live closures, construction, or traffic.

## Layout

```
pipeline/
  config.py           AOI, CRS, resolution, endpoints
  fetch_dem.py        Copernicus GLO-30 → dem.tif
  fetch_osm.py        Overpass → roads / waterways / water
  fetch_rainfall.py   Open-Meteo → rainfall.json + trigger series
  build_hydrology.py  priority-flood, D8, TWI, HAND
  build_network.py    OSM ways → routable graph + per-edge features
  score_risk.py       the weighted-overlay index (weights live here)
  build_checkset.py   independent check locations via Nominatim
  validate.py         terrain cross-check + locality check-set
  validate_bbmp.py    check against this repo's BBMP points
  fit_supervised.py   slot for real labels — not yet usable
  routing.py          flood-aware Dijkstra, fastest vs safest
serve.py              stdlib HTTP server
web/index.html        Leaflet UI
src/                  original trial model — superseded, see issues above
```

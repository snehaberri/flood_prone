"""Shared configuration for the Bengaluru flood-risk pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
WEB = ROOT / "web"
for _d in (RAW, PROC, WEB):
    _d.mkdir(parents=True, exist_ok=True)

# BBMP / Greater Bengaluru core. Generous enough to cover the outer ring road.
BBOX = dict(south=12.80, west=77.44, north=13.15, east=77.78)
BBOX_TUPLE = (BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])

# Metric CRS for all area/distance work. UTM 43N covers Bengaluru.
CRS_M = "EPSG:32643"
CRS_WGS = "EPSG:4326"

# Working raster resolution, metres. The DEM is 30 m native; we resample to 20 m
# so the hydrology rasters line up on a clean grid.
PIXEL_M = 20

COP_DEM_TILES = ["N12_00_E077_00", "N13_00_E077_00"]
COP_DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif"
)

OVERPASS = "https://overpass-api.de/api/interpreter"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# Centre of the city, used as the map's default view.
CITY_CENTRE = (12.9716, 77.5946)

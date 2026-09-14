"""Pull the road network, waterways and water bodies for the AOI from Overpass.

The road layer is fetched as a grid of sub-tiles rather than one request. A single
bbox query for the whole BBMP drivable network reliably 504s on the public Overpass
instances - the response is too large to build inside the server timeout. Tiling
keeps each request small, lets a single failure be retried in isolation, and caches
per tile so an interrupted run resumes where it stopped.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# Drivable network. Service/track roads are excluded: they roughly double the edge
# count and are not roads anyone routes a car down.
ROAD_RE = "^(motorway|trunk|primary|secondary|tertiary|unclassified|residential)(_link)?$"

ROAD_TILES = 4          # 4x4 = 16 sub-requests for the road layer
UA = {"User-Agent": "flood-risk-blr/1.0 (student research)"}


def _post(query: str, label: str, tries: int = 6) -> dict:
    """POST an Overpass query, retrying across mirrors.

    An empty result is treated as a failure worth retrying. Overpass sometimes
    answers a query it could not complete with 200 and zero elements rather than an
    error, and caching that as success silently deletes a whole region from the map -
    which is exactly how the lake layer lost Hebbal on the first run. Only after every
    mirror has returned empty do we accept that the area really is empty, and say so.
    """
    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for attempt in range(tries):
        url = MIRRORS[attempt % len(MIRRORS)]
        try:
            print(f"    [{label}] {url.split('/')[2]} try {attempt + 1} ...", flush=True)
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=420) as r:
                raw = r.read()
            js = json.loads(raw)
            if "elements" not in js:
                raise ValueError("no elements key in response")
            if not js["elements"] and attempt < tries - 1:
                raise ValueError("zero elements - suspect a silent server timeout")
            if not js["elements"]:
                print(f"    [{label}] WARNING: every mirror returned empty; "
                      f"treating this area as genuinely having no matching features")
            return js
        except Exception as e:                          # noqa: BLE001 - report and retry
            last = e
            print(f"    [{label}] {type(e).__name__}: {str(e)[:110]}")
            time.sleep(min(30, 8 * (attempt + 1)))
    raise SystemExit(f"{label}: all Overpass attempts failed - last error {last}")


def _cached(name: str):
    p = C.RAW / f"osm_{name}.json"
    if p.exists() and p.stat().st_size > 500:
        js = json.loads(p.read_text(encoding="utf-8"))
        print(f"  cached  {name}: {len(js['elements'])} elements")
        return js
    return None


def _save(name: str, js: dict):
    p = C.RAW / f"osm_{name}.json"
    p.write_text(json.dumps(js), encoding="utf-8")
    print(f"  wrote   {name}: {len(js['elements'])} elements, {p.stat().st_size / 1e6:.1f} MB")


def fetch_tiled(name: str, body_template: str, tiles: int):
    """Fetch a layer as a grid of sub-requests, caching each tile separately.

    `body_template` must contain a single {b} placeholder for the bbox string.
    Ways are returned whole by Overpass even when they straddle a tile edge, so
    tiles are de-duplicated by way id after the fact.
    """
    if (js := _cached(name)) is not None:
        return js

    s, w = C.BBOX["south"], C.BBOX["west"]
    dy = (C.BBOX["north"] - s) / tiles
    dx = (C.BBOX["east"] - w) / tiles

    seen, elements = set(), []
    for r in range(tiles):
        for c in range(tiles):
            tag = f"{name}_{r}{c}"
            tile = _cached(tag)
            if tile is None:
                b = f"{s + r * dy},{w + c * dx},{s + (r + 1) * dy},{w + (c + 1) * dx}"
                tile = _post(body_template.format(b=b), tag)
                _save(tag, tile)
                time.sleep(3)          # be a good citizen on a free shared service
            for el in tile["elements"]:
                if el["id"] not in seen:
                    seen.add(el["id"])
                    elements.append(el)

    js = {"elements": elements}
    _save(name, js)
    return js


def main():
    print("roads:")
    fetch_tiled("roads",
                '[out:json][timeout:300];'
                f'(way["highway"~"{ROAD_RE}"]["area"!~"yes"]({{b}}););'
                'out body geom;', ROAD_TILES)

    print("waterways:")
    # Bengaluru storm water drains are tagged inconsistently - rajakaluves appear as
    # drain, ditch, stream and canal depending on the mapper. Take all of them.
    fetch_tiled("waterways",
                '[out:json][timeout:300];'
                '(way["waterway"~"^(river|stream|canal|drain|ditch)$"]({b}););'
                'out body geom;', 2)

    print("water bodies:")
    # The kere/tank network. Tiled at 3x3 because the combined water query reliably
    # 504s over the whole AOI.
    fetch_tiled("water",
                '[out:json][timeout:300];'
                '(way["natural"="water"]({b});'
                ' way["landuse"="reservoir"]({b}););'
                'out body geom;', 3)

    print("\n  all OSM layers present in data/raw/")


if __name__ == "__main__":
    main()

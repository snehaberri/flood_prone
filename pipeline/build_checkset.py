"""Build an INDEPENDENT check-set of documented chronic flooding locations.

Provenance and limits - read before using any number this produces
-----------------------------------------------------------------
These are Bengaluru localities repeatedly named in public reporting on the
September 2022 and 2024 urban floods, and in BBMP's own flood-prone-area
statements. They are compiled from public reporting, not from a measurement
dataset. Specifically:

  * There are no dates attached. A locality appears because it floods often,
    not because of a recorded event.
  * Nominatim returns a *locality centroid*. The actual flooding is at junction
    and street scale - often a single underpass. A centroid can easily sit a few
    hundred metres from the road that actually floods.
  * The set is small (tens of points) and geographically biased toward the
    south-east, because that is where the reporting concentrated.

Consequently this is a SANITY CHECK, not a test set, and nothing here is used to
choose or tune the model weights - those are fixed a priori in score_risk.py.
Treat a good result as "the index is not obviously wrong" and a bad result as a
reason to investigate, not as an accuracy figure to quote.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

NOMINATIM = "https://nominatim.openstreetmap.org/search"

# Localities and junctions repeatedly named in flood reporting for Bengaluru.
LOCATIONS = [
    "Silk Board Junction, Bengaluru",
    "Bellandur, Bengaluru",
    "Varthur, Bengaluru",
    "Sarjapur Road, Bengaluru",
    "Koramangala, Bengaluru",
    "Ejipura, Bengaluru",
    "HSR Layout, Bengaluru",
    "Bommanahalli, Bengaluru",
    "Madiwala, Bengaluru",
    "Marathahalli, Bengaluru",
    "Mahadevapura, Bengaluru",
    "Doddanekkundi, Bengaluru",
    "Whitefield, Bengaluru",
    "KR Puram, Bengaluru",
    "Hebbal, Bengaluru",
    "Nagawara, Bengaluru",
    "Thanisandra, Bengaluru",
    "Horamavu, Bengaluru",
    "Yelahanka, Bengaluru",
    "Okalipuram, Bengaluru",
    "Nayandahalli, Bengaluru",
    "Hosakerehalli, Bengaluru",
    "Sumanahalli, Bengaluru",
    "Rajajinagar, Bengaluru",
    "Shantinagar, Bengaluru",
    "KR Market, Bengaluru",
    "Yemalur, Bengaluru",
    "Kaikondrahalli, Bengaluru",
    "Kasavanahalli, Bengaluru",
    "Agara, Bengaluru",
]

# Control set: localities on higher ground that do NOT appear in the flood
# reporting. Without these, "everywhere scores high" would look like success.
CONTROLS = [
    "Malleshwaram, Bengaluru",
    "Basavanagudi, Bengaluru",
    "Jayanagar, Bengaluru",
    "Sadashivanagar, Bengaluru",
    "Vidyaranyapura, Bengaluru",
    "Banashankari, Bengaluru",
    "Kumaraswamy Layout, Bengaluru",
    "Dollars Colony, Bengaluru",
    "RT Nagar, Bengaluru",
    "Girinagar, Bengaluru",
]


def geocode(q: str):
    params = {"q": q, "format": "json", "limit": 1,
              "viewbox": f"{C.BBOX['west']},{C.BBOX['north']},{C.BBOX['east']},{C.BBOX['south']}",
              "bounded": 1}
    url = f"{NOMINATIM}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "flood-risk-blr/1.0 (student research; contact via repo)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            js = json.loads(r.read())
    except Exception as e:                                  # noqa: BLE001
        print(f"    ! {q}: {type(e).__name__}")
        return None
    if not js:
        print(f"    ? {q}: no match")
        return None
    return float(js[0]["lat"]), float(js[0]["lon"])


def main():
    dest = C.PROC / "checkset.json"
    if dest.exists():
        print(f"  cached {dest}")
        return

    out = []
    for label, names in (("flood_prone", LOCATIONS), ("control", CONTROLS)):
        print(f"  {label}:")
        for q in names:
            hit = geocode(q)
            time.sleep(1.1)                  # Nominatim policy: max 1 request/second
            if hit is None:
                continue
            lat, lon = hit
            out.append({"name": q.replace(", Bengaluru", ""), "label": label,
                        "lat": lat, "lon": lon})
            print(f"    {q.replace(', Bengaluru', ''):<24} {lat:.4f}, {lon:.4f}")

    n_f = sum(1 for o in out if o["label"] == "flood_prone")
    n_c = len(out) - n_f
    dest.write_text(json.dumps({
        "source": "Localities named in public reporting on the 2022/2024 Bengaluru "
                  "floods; coordinates are OSM/Nominatim locality centroids.",
        "caveat": "Sanity check only - no dates, centroid-level precision, small and "
                  "geographically biased. Not used to fit or tune the model.",
        "points": out,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {dest}: {n_f} flood-prone, {n_c} control")


if __name__ == "__main__":
    main()

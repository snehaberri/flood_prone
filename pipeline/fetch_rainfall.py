"""Fetch hourly rainfall for Bengaluru from Open-Meteo and derive a flood trigger.

Open-Meteo is free, needs no API key, and serves both recent past and forecast hours
in one call. This is the *time* axis of the model: terrain says where water collects,
rainfall says whether there is enough water to collect.

We sample a grid across the city rather than one point. Bengaluru convective storms
are famously local - the classic pattern is 80 mm over Bellandur while Malleshwaram
stays dry - so a single city-centre value would be wrong most of the time. The
underlying weather model is ~11 km, so a 4x4 grid is about as fine as is meaningful.

Trigger thresholds follow IMD rainfall categories:
    24 h  64.5 mm = "heavy", 115.6 mm = "very heavy"
Short-duration thresholds are the ones that matter for urban pluvial flooding, where
drains are overwhelmed in under an hour.
"""
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C

GRID = 4  # 4x4 sample points across the AOI

# Rainfall accumulation at which the trigger saturates (= 1.0), in mm.
TRIGGER_SCALES = {"rain_1h": 25.0, "rain_3h": 40.0, "rain_24h": 80.0}


def sample_points():
    s, w = C.BBOX["south"], C.BBOX["west"]
    dy = (C.BBOX["north"] - s) / GRID
    dx = (C.BBOX["east"] - w) / GRID
    return [(round(s + (r + 0.5) * dy, 4), round(w + (c + 0.5) * dx, 4))
            for r in range(GRID) for c in range(GRID)]


def fetch(past_days: int = 3, forecast_days: int = 3) -> dict:
    pts = sample_points()
    params = {
        "latitude": ",".join(str(p[0]) for p in pts),
        "longitude": ",".join(str(p[1]) for p in pts),
        "hourly": "precipitation",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }
    url = f"{C.OPEN_METEO}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "flood-risk-blr/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        js = json.loads(r.read())
    # Open-Meteo returns a bare dict for one point, a list for many.
    return js if isinstance(js, list) else [js]


def rolling(series, hours):
    """Trailing sum over `hours`, aligned so index i covers (i-hours, i]."""
    a = np.asarray(series, dtype="float64")
    a = np.nan_to_num(a)
    cs = np.concatenate([[0.0], np.cumsum(a)])
    out = np.full(a.size, np.nan)
    for i in range(a.size):
        lo = max(0, i + 1 - hours)
        out[i] = cs[i + 1] - cs[lo]
    return out


def build():
    blocks = fetch()
    times = blocks[0]["hourly"]["time"]
    n = len(times)

    precip = np.array([b["hourly"]["precipitation"] for b in blocks], dtype="float64")
    precip = np.nan_to_num(precip)                               # (points, hours)

    pts = [(b["latitude"], b["longitude"]) for b in blocks]

    per_point = []
    for k, (lat, lon) in enumerate(pts):
        per_point.append({
            "lat": lat, "lon": lon,
            "rain_1h": precip[k].tolist(),
            "rain_3h": rolling(precip[k], 3).tolist(),
            "rain_24h": rolling(precip[k], 24).tolist(),
        })

    # City-wide trigger per hour: the worst point, because a route crossing the city
    # is exposed to the wettest cell it passes through, not the average.
    trig = np.zeros(n)
    for p in per_point:
        for key, scale in TRIGGER_SCALES.items():
            trig = np.maximum(trig, np.clip(np.array(p[key]) / scale, 0, 1))

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timezone": "Asia/Kolkata",
        "times": times,
        "trigger_scales_mm": TRIGGER_SCALES,
        "points": per_point,
        "city_trigger": np.round(trig, 4).tolist(),
    }
    dest = C.PROC / "rainfall.json"
    dest.write_text(json.dumps(out), encoding="utf-8")

    peak = int(np.argmax(trig))
    print(f"  {len(pts)} grid points x {n} hours ({times[0]} .. {times[-1]})")
    print(f"  total rain over window, city max : {precip.sum(axis=1).max():.1f} mm")
    print(f"  peak trigger {trig[peak]:.2f} at {times[peak]}")
    print(f"  wrote {dest}")
    return out


if __name__ == "__main__":
    build()

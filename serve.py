"""Minimal stdlib HTTP server for the flood-risk map.

No web framework on purpose - http.server is enough for a local research tool and
keeps the dependency list to the scientific stack. Run:

    python serve.py            then open http://127.0.0.1:8000

Endpoints
    GET /api/meta                      model weights, sensitivity, validation summary
    GET /api/rainfall                  hourly rainfall + trigger series
    GET /api/risk?min=0.6              GeoJSON of edges at or above a susceptibility
    GET /api/route?o=lat,lon&d=lat,lon&trigger=0.8
"""
import json
import sys
import threading
import urllib.parse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pipeline"))
import config as C
from routing import Router

ROUTER = None
_LOCK = threading.Lock()


def router():
    global ROUTER
    with _LOCK:
        if ROUTER is None:
            ROUTER = Router()
    return ROUTER


# Below this length, an edge's interior vertices are sub-pixel at city zoom, so we
# send only its endpoints. The median edge is ~43 m, so this removes most of the
# payload without any visible change to the drawn network.
THIN_BELOW_M = 120.0


@lru_cache(maxsize=16)
def risk_geojson(min_score: float) -> bytes:
    r = router()
    e = r.edges
    sel = e[e.susceptibility >= min_score]
    feats = []
    for coords, s, nm, hw, ln in zip(sel.coords, sel.susceptibility, sel.name,
                                     sel.highway, sel.length_m):
        pts = json.loads(coords)
        if ln < THIN_BELOW_M and len(pts) > 2:
            pts = [pts[0], pts[-1]]
        feats.append({
            "type": "Feature",
            "properties": {"s": round(float(s), 3), "name": nm or "", "hw": hw},
            "geometry": {"type": "LineString",
                         "coordinates": [[round(x, 5), round(y, 5)] for x, y in pts]},
        })
    return json.dumps({"type": "FeatureCollection", "features": feats}).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # keep the console readable
        pass

    def _send(self, body: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, code, msg):
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        p = u.path

        try:
            if p in ("/", "/index.html"):
                return self._send((C.WEB / "index.html").read_bytes(), "text/html; charset=utf-8")

            if p == "/api/meta":
                out = {}
                for key, fname in [("model", "model_meta.json"),
                                   ("validation", "validation.json")]:
                    f = C.PROC / fname
                    out[key] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
                return self._send(json.dumps(out).encode())

            if p == "/api/rainfall":
                f = C.PROC / "rainfall.json"
                if not f.exists():
                    return self._fail(404, "rainfall.json missing - run fetch_rainfall.py")
                return self._send(f.read_bytes())

            if p == "/api/risk":
                m = float(q.get("min", ["0.6"])[0])
                return self._send(risk_geojson(round(max(0.0, min(1.0, m)), 2)))

            if p == "/api/route":
                try:
                    o = [float(x) for x in q["o"][0].split(",")]
                    d = [float(x) for x in q["d"][0].split(",")]
                except (KeyError, ValueError):
                    return self._fail(400, "need o=lat,lon and d=lat,lon")
                t = float(q.get("trigger", ["0"])[0])
                res = router().route(o[0], o[1], d[0], d[1], trigger=max(0.0, min(1.0, t)))
                return self._send(json.dumps(res).encode())

            return self._fail(404, f"no route {p}")
        except Exception as e:                               # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._fail(500, f"{type(e).__name__}: {e}")


def main(port=8000):
    print("  loading graph ...", flush=True)
    router()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    # flush: serve_forever() blocks immediately after this, so an unflushed line
    # sits in the buffer and anything tailing the log never sees the server come up.
    print(f"\n  ready -> http://127.0.0.1:{port}\n  ctrl-c to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)

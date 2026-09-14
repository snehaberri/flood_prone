"""Flood-aware routing over the scored road graph.

Two routes are always computed for the same origin/destination:

  fastest  - ordinary shortest travel time, flood risk ignored
  safest   - travel time inflated by flood risk, and edges above the impassable
             threshold removed from the graph entirely

Returning both is the point. A flood-aware route with nothing to compare against
tells the user nothing about what avoiding the water actually cost them.

Cost model
----------
    risk_e(t)  = susceptibility_e * trigger(t)
    cost_e(t)  = base_time_e * (1 + PENALTY * risk_e(t))

`trigger` is the rainfall term from fetch_rainfall.py, 0 (dry) to 1 (saturating
downpour). With no rain every risk term is 0 and the two routes coincide - which is
the correct behaviour, not a bug.
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from score_risk import IMPASSABLE_RISK

# How much a fully-flooded road is penalised. 8.0 means such a road is treated as
# nine times its free-flow travel time - strongly avoided, but still usable if it is
# the only way through, which matters for not stranding anyone on an island.
PENALTY = 8.0


class Router:
    def __init__(self):
        e = pd.read_parquet(C.PROC / "edges_scored.parquet")
        n = pd.read_parquet(C.PROC / "nodes.parquet")

        # Compact node ids to contiguous integers for fast array indexing.
        used = pd.unique(pd.concat([e.u, e.v]))
        self.node_ids = used
        idx = {nid: i for i, nid in enumerate(used)}
        coords = n.set_index("node").loc[:, ["lon", "lat"]]
        ll = coords.reindex(used)
        self.lon = ll.lon.to_numpy()
        self.lat = ll.lat.to_numpy()

        e = e.assign(ui=e.u.map(idx).to_numpy(), vi=e.v.map(idx).to_numpy())
        self.edges = e

        # Adjacency as lists of (neighbour, edge_row). Two-way unless oneway.
        self.adj = [[] for _ in range(len(used))]
        for row, (ui, vi, ow) in enumerate(zip(e.ui, e.vi, e.oneway)):
            self.adj[ui].append((vi, row))
            if not ow:
                self.adj[vi].append((ui, row))

        ok = np.isfinite(self.lon) & np.isfinite(self.lat)
        self._ok_idx = np.flatnonzero(ok)
        # Equirectangular projection is plenty accurate for nearest-node lookup
        # over a 40 km city and avoids a pyproj call per query.
        k = np.cos(np.radians(C.CITY_CENTRE[0]))
        self.tree = cKDTree(np.c_[self.lon[ok] * k, self.lat[ok]])
        self._k = k

        self.base_time = e.base_time_s.to_numpy()
        self.susc = e.susceptibility.to_numpy()
        self.length = e.length_m.to_numpy()
        print(f"  router: {len(used):,} nodes, {len(e):,} edges")

    def nearest_node(self, lat, lon):
        _, i = self.tree.query([lon * self._k, lat])
        return int(self._ok_idx[i])

    def _costs(self, trigger):
        risk = self.susc * float(trigger)
        cost = self.base_time * (1.0 + PENALTY * risk)
        blocked = risk >= IMPASSABLE_RISK
        return cost, risk, blocked

    def _dijkstra(self, src, dst, cost, blocked=None):
        n = len(self.adj)
        dist = np.full(n, np.inf)
        prev_node = np.full(n, -1, dtype=np.int64)
        prev_edge = np.full(n, -1, dtype=np.int64)
        dist[src] = 0.0
        pq = [(0.0, src)]
        seen = np.zeros(n, bool)
        while pq:
            d, u = heapq.heappop(pq)
            if seen[u]:
                continue
            seen[u] = True
            if u == dst:
                break
            for v, row in self.adj[u]:
                if seen[v] or (blocked is not None and blocked[row]):
                    continue
                nd = d + cost[row]
                if nd < dist[v]:
                    dist[v] = nd
                    prev_node[v] = u
                    prev_edge[v] = row
                    heapq.heappush(pq, (nd, v))
        if not np.isfinite(dist[dst]):
            return None
        # Keep the node we arrived *from* alongside each edge. An edge is stored in
        # u->v order but may be walked either way, and a segment appended in the wrong
        # direction turns the drawn route into a zigzag.
        steps, u = [], dst
        while u != src:
            steps.append((int(prev_edge[u]), int(prev_node[u])))
            u = int(prev_node[u])
        return steps[::-1]

    def _summarise(self, steps, risk, label):
        if steps is None:
            return None
        rows = np.asarray([s[0] for s in steps])
        ui = self.edges.ui.to_numpy()
        coords = []
        worst_joint = 0.0
        k = np.cos(np.radians(C.CITY_CENTRE[0]))
        for row, from_node in steps:
            seg = json.loads(self.edges.coords.iat[int(row)])
            if ui[row] != from_node:          # walked v -> u, so flip the geometry
                seg = seg[::-1]
            if coords:
                # Consecutive edges share a junction node, so this distance must be
                # ~0. Anything larger means a segment went in backwards. Measured at
                # the joints only - vertex spacing *within* an OSM way is legitimately
                # hundreds of metres on a straight flyover.
                a, b = coords[-1], seg[0]
                worst_joint = max(worst_joint, float(np.hypot(
                    (a[0] - b[0]) * k * 111_320, (a[1] - b[1]) * 111_320)))
            # Drop the duplicated joint between consecutive segments.
            coords.extend(seg if not coords else seg[1:])
        d = float(self.length[rows].sum())
        t = float(self.base_time[rows].sum())
        rk = risk[rows]
        w = self.length[rows]
        return {
            "label": label,
            "distance_m": round(d, 1),
            "free_flow_time_s": round(t, 1),
            "mean_risk": round(float(np.average(rk, weights=w)) if w.sum() else 0.0, 4),
            "max_risk": round(float(rk.max()) if len(rk) else 0.0, 4),
            "high_risk_m": round(float(w[rk >= 0.4].sum()), 1),
            "n_edges": int(len(rows)),
            "joint_gap_m": round(worst_joint, 2),
            "geometry": coords,
        }

    def route(self, o_lat, o_lon, d_lat, d_lon, trigger=0.0):
        src = self.nearest_node(o_lat, o_lon)
        dst = self.nearest_node(d_lat, d_lon)
        if src == dst:
            return {"error": "origin and destination snap to the same junction"}

        cost, risk, blocked = self._costs(trigger)

        fastest = self._summarise(self._dijkstra(src, dst, self.base_time), risk, "fastest")
        safest = self._summarise(self._dijkstra(src, dst, cost, blocked), risk, "safest")
        if safest is None:
            # Every path crosses a blocked road. Fall back to soft avoidance so the
            # user gets a route plus an explicit warning, rather than nothing.
            safest = self._summarise(self._dijkstra(src, dst, cost), risk, "safest")
            if safest:
                safest["warning"] = ("every available route crosses a road expected to be "
                                     "impassable; this is the least-bad option")
        out = {
            "trigger": round(float(trigger), 3),
            "blocked_edges": int(blocked.sum()),
            "origin_snap": [self.lat[src], self.lon[src]],
            "dest_snap": [self.lat[dst], self.lon[dst]],
            "routes": [r for r in (fastest, safest) if r],
        }
        if fastest and safest:
            out["comparison"] = {
                "extra_distance_m": round(safest["distance_m"] - fastest["distance_m"], 1),
                "extra_time_s": round(safest["free_flow_time_s"] - fastest["free_flow_time_s"], 1),
                "risk_avoided": round(fastest["mean_risk"] - safest["mean_risk"], 4),
                "identical": fastest["geometry"] == safest["geometry"],
            }
        return out


if __name__ == "__main__":
    r = Router()
    # Silk Board -> Hebbal, the canonical Bengaluru monsoon nightmare.
    res = r.route(12.9166, 77.6229, 13.0358, 77.5970, trigger=0.8)
    print(json.dumps({k: v for k, v in res.items() if k != "routes"}, indent=2))
    for rt in res["routes"]:
        print(f"  {rt['label']:<8} {rt['distance_m']/1000:6.2f} km  "
              f"{rt['free_flow_time_s']/60:5.1f} min  "
              f"mean risk {rt['mean_risk']:.3f}  max {rt['max_risk']:.3f}  "
              f"high-risk {rt['high_risk_m']/1000:.2f} km  "
              f"joint gap {rt['joint_gap_m']:.2f} m")
        if rt["joint_gap_m"] > 1.0:
            print("    ^ WARNING: segments are not joining - check orientation "
                  "in _summarise()")

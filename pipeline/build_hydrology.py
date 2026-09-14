"""Derive the terrain-hydrology rasters that drive the susceptibility model.

Outputs (all aligned to data/processed/dem.tif):
    sink_depth.tif  depth of each closed depression, metres    - where water ponds
    slope.tif       slope in degrees                           - flat land drains slowly
    flowacc.tif     D8 upslope contributing cells              - where water converges
    twi.tif         topographic wetness index                  - standard saturation proxy
    hand.tif        height above nearest drainage, metres      - low = at drainage level

Implemented directly on numpy rather than via richdem/pysheds, which have no
Python 3.14 wheels. The algorithms are the standard ones:
  - depression filling: Priority-Flood with an epsilon gradient (Barnes et al. 2014)
  - flow routing: D8 steepest descent (OCallaghan & Mark 1984)
  - HAND: Renno et al. 2008, drainage defined by a flow-accumulation threshold

A note on the DEM. Copernicus GLO-30 is a *surface* model: it includes rooftops and
tree canopy. For pluvial flooding we want bare ground. A full DSM-to-DTM inversion is
out of scope, so we apply a grey-scale morphological opening at ~60 m, which removes
building-scale positive spikes while leaving the valley structure intact. This is an
approximation and is the single largest source of error in the terrain features.
"""
import heapq
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# A cell needs this many upslope cells before it counts as a drainage channel.
# 200 cells at 20 m = 8 ha, which picks out the rajakaluve-scale valley network
# rather than every garden swale.
DRAINAGE_THRESHOLD_CELLS = 200

# D8 neighbour offsets.
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _read_dem():
    with rasterio.open(C.PROC / "dem.tif") as s:
        return s.read(1).astype("float64"), s.profile


def _write(name, arr, profile, dtype="float32"):
    p = dict(profile, dtype=dtype, count=1, compress="deflate", nodata=np.nan)
    with rasterio.open(C.PROC / name, "w", **p) as d:
        d.write(arr.astype(dtype), 1)
    print(f"  wrote  {name}")


def despike(dem):
    """Strip building/canopy-scale positive spikes from the DSM."""
    nan = ~np.isfinite(dem)
    work = dem.copy()
    # Fill nodata by nearest neighbour first, so morphology does not eat the footprint.
    if nan.any():
        idx = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
        work = work[tuple(idx)]
    opened = ndimage.grey_opening(work, size=3)       # 3 px @ 20 m = 60 m
    opened = ndimage.median_filter(opened, size=3)
    opened[nan] = np.nan
    return opened


def fill_depressions(dem):
    """Priority-Flood with epsilon gradient. Returns the filled surface."""
    ny, nx = dem.shape
    filled = dem.copy()
    valid = np.isfinite(dem)
    closed = ~valid.copy()

    # Seed: every valid cell on the grid edge, plus every valid cell touching nodata.
    edge = np.zeros_like(valid)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    if (~valid).any():
        edge |= ndimage.binary_dilation(~valid, structure=np.ones((3, 3), bool)) & valid
    edge &= valid

    seeds = np.argwhere(edge)
    heap = [(float(dem[i, j]), int(i), int(j)) for i, j in seeds]
    heapq.heapify(heap)
    closed[edge] = True

    eps = 1e-4
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        z, i, j = pop(heap)
        for di, dj in NB:
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or ni >= ny or nj >= nx or closed[ni, nj]:
                continue
            closed[ni, nj] = True
            zn = filled[ni, nj]
            if zn <= z:
                zn = z + eps
                filled[ni, nj] = zn
            push(heap, (float(zn), ni, nj))
    return filled


def d8_receivers(filled):
    """Flat index of the neighbour each cell drains to (-1 if it drains nowhere)."""
    ny, nx = filled.shape
    best_drop = np.full(filled.shape, -np.inf)
    recv = np.full(filled.shape, -1, dtype=np.int64)
    z = np.where(np.isfinite(filled), filled, np.inf)

    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    for di, dj in NB:
        dist = np.hypot(di, dj) * C.PIXEL_M
        # shifted[i, j] must hold z[i + di, j + dj] - the elevation of the very
        # neighbour whose index we record in `recv` below. Getting these two out of
        # step silently produces valid-looking but meaningless flow directions.
        shifted = np.full_like(z, np.inf)
        shifted[max(0, -di):ny - max(0, di), max(0, -dj):nx - max(0, dj)] = \
            z[max(0, di):ny + min(0, di), max(0, dj):nx + min(0, dj)]
        with np.errstate(invalid="ignore"):      # inf - inf at nodata, filtered below
            drop = (z - shifted) / dist
        ni, nj = ii + di, jj + dj
        inb = (ni >= 0) & (ni < ny) & (nj >= 0) & (nj < nx)
        better = (drop > best_drop) & inb & np.isfinite(shifted) & np.isfinite(drop)
        best_drop = np.where(better, drop, best_drop)
        recv = np.where(better, ni * nx + nj, recv)

    recv[~np.isfinite(filled)] = -1
    recv[best_drop <= 0] = -1
    return recv


def flow_accumulation(filled, recv):
    """Cell counts accumulated downstream, in descending-elevation order."""
    z = filled.ravel()
    r = recv.ravel()
    acc = np.where(np.isfinite(z), 1.0, 0.0)
    order = np.argsort(-np.where(np.isfinite(z), z, -np.inf), kind="stable")
    for idx in order:
        d = r[idx]
        if d >= 0:
            acc[d] += acc[idx]
    return acc.reshape(filled.shape)


def hand(filled, recv, drainage):
    """Height above nearest drainage, by ascending-elevation propagation.

    After filling, every receiver is strictly lower than its cell, so processing in
    ascending elevation guarantees a cell's receiver already carries its value.
    """
    z = filled.ravel()
    r = recv.ravel()
    drn = drainage.ravel()
    nd = np.full(z.size, np.nan)
    order = np.argsort(np.where(np.isfinite(z), z, np.inf), kind="stable")
    for idx in order:
        if not np.isfinite(z[idx]):
            continue
        if drn[idx]:
            nd[idx] = z[idx]
        else:
            d = r[idx]
            nd[idx] = nd[d] if (d >= 0 and np.isfinite(nd[d])) else z[idx]
    return (z - nd).reshape(filled.shape)


def main():
    dem, profile = _read_dem()
    print(f"  dem    {dem.shape}  valid {np.isfinite(dem).sum():,}")

    print("  despiking DSM ...", flush=True)
    sm = despike(dem)

    print("  slope ...", flush=True)
    gy, gx = np.gradient(np.nan_to_num(sm, nan=float(np.nanmean(sm))), C.PIXEL_M)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~np.isfinite(sm)] = np.nan

    print("  filling depressions (priority-flood) ...", flush=True)
    filled = fill_depressions(sm)
    sink = filled - sm
    sink[~np.isfinite(sm)] = np.nan
    print(f"         max sink {np.nanmax(sink):.2f} m, "
          f"{int((sink > 0.1).sum()):,} cells ponding >0.1 m")

    print("  D8 flow directions ...", flush=True)
    recv = d8_receivers(filled)

    print("  flow accumulation ...", flush=True)
    acc = flow_accumulation(filled, recv)
    n_valid = int(np.isfinite(sm).sum())
    peak = float(np.nanmax(acc))
    print(f"         max accumulation {peak:,.0f} cells "
          f"({peak / n_valid:.1%} of the grid)")
    # On a correctly routed DEM the trunk valleys drain a large share of the domain.
    # A tiny peak means flow directions are inconsistent and every downstream
    # product - TWI, HAND, the whole index - is quietly meaningless.
    if peak < 0.01 * n_valid:
        raise SystemExit(
            f"flow routing looks broken: peak accumulation {peak:,.0f} is under 1% of "
            f"{n_valid:,} valid cells. Expected the main drainage lines to collect "
            f"tens of thousands of cells. Check d8_receivers()."
        )

    print("  TWI ...", flush=True)
    spec_area = acc * C.PIXEL_M
    twi = np.log(np.maximum(spec_area, 1.0) /
                 np.maximum(np.tan(np.radians(np.nan_to_num(slope))), 0.001))
    twi[~np.isfinite(sm)] = np.nan

    print("  HAND ...", flush=True)
    drainage = acc >= DRAINAGE_THRESHOLD_CELLS
    print(f"         drainage network {int(drainage.sum()):,} cells")
    h = np.clip(hand(filled, recv, drainage), 0, None)
    h[~np.isfinite(sm)] = np.nan

    for name, arr in [("sink_depth.tif", sink), ("slope.tif", slope),
                      ("flowacc.tif", acc), ("twi.tif", twi), ("hand.tif", h),
                      ("dem_smooth.tif", sm),
                      ("drainage.tif", drainage.astype("float32"))]:
        _write(name, arr, profile)
    print("\n  hydrology complete")


if __name__ == "__main__":
    main()

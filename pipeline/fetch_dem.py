"""Download Copernicus GLO-30 tiles, merge, reproject to UTM 43N, clip to the AOI.

Copernicus DEM is a DSM: it includes buildings and tree canopy. For urban pluvial
work we want bare ground, so `build_hydrology.py` applies a morphological opening
to strip building-scale spikes. See the note there.
"""
import sys, urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name}")
        return dest
    print(f"  fetching {dest.name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "flood-risk-blr/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"  got     {dest.name}  {dest.stat().st_size/1e6:.1f} MB")
    return dest


def main() -> None:
    tiles = [download(C.COP_DEM_URL.format(t=t), C.RAW / f"cop30_{t}.tif")
             for t in C.COP_DEM_TILES]

    srcs = [rasterio.open(t) for t in tiles]
    mosaic, transform = merge(srcs, bounds=C.BBOX_TUPLE)
    meta = srcs[0].meta.copy()
    src_crs = srcs[0].crs
    for s in srcs:
        s.close()
    print(f"  merged  {mosaic.shape} in {src_crs}")

    # Reproject to metric CRS at PIXEL_M so slope/flow are in real units.
    dst_transform, w, h = calculate_default_transform(
        src_crs, C.CRS_M, mosaic.shape[2], mosaic.shape[1],
        *rasterio.transform.array_bounds(mosaic.shape[1], mosaic.shape[2], transform),
        resolution=C.PIXEL_M,
    )
    # Init to NaN, not zero. A lat/lon rectangle is not a rectangle in UTM, so the
    # corners of the destination grid fall outside the source tiles. The Copernicus
    # COGs declare nodata=None, so those pixels would otherwise stay 0 - which the
    # hydrology stage reads as a 900 m deep pit and scores as extreme flood risk.
    dst = np.full((h, w), np.nan, dtype="float32")
    reproject(
        source=mosaic[0], destination=dst,
        src_transform=transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=C.CRS_M,
        resampling=Resampling.bilinear, src_nodata=meta.get("nodata"), dst_nodata=np.nan,
        init_dest_nodata=False,
    )

    # Defensive plausibility guard. The Bengaluru plateau sits at roughly 740-1000 m;
    # anything far outside that is an edge or void artefact, not terrain.
    implausible = np.isfinite(dst) & ((dst < 700) | (dst > 1200))
    if implausible.any():
        print(f"  masked  {implausible.sum()} implausible pixels (outside 700-1200 m)")
        dst[implausible] = np.nan

    out = C.PROC / "dem.tif"
    with rasterio.open(
        out, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=C.CRS_M, transform=dst_transform, nodata=np.nan, compress="deflate",
    ) as d:
        d.write(dst, 1)

    valid = dst[np.isfinite(dst)]
    print(f"\n  wrote   {out}")
    print(f"  grid    {w} x {h} @ {C.PIXEL_M} m")
    print(f"  elev    {valid.min():.1f} - {valid.max():.1f} m  (mean {valid.mean():.1f})")


if __name__ == "__main__":
    main()

"""Shared I/O, logging, grid, and raster helpers for the HEC pipeline.

Every module imports from this file so that CRS handling, GeoTIFF writing,
and the master grid definition are guaranteed to be identical across the
whole pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    """Emit a pipeline-tagged message to stdout (flushed for live progress)."""
    print(f"[HEP] {msg}", flush=True)


def section(title: str) -> None:
    """Print a clearly delimited section header in the log stream."""
    log("=" * 78)
    log(title)
    log("=" * 78)


# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------
def layer_exists(path: Optional[str]) -> bool:
    """Return True when *path* is set, points to a file, and is non-empty."""
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0


# -----------------------------------------------------------------------------
# Output folder routing
# -----------------------------------------------------------------------------
# Each pipeline stage writes into its own numbered subfolder under outputs/.
# Numbering matches pipeline order so browsing outputs/ in Finder, QGIS, or
# a shell shows the stages in the order they run.
_MODULE_FOLDERS = {
    "density":   "01_density",
    "hotspots":  "02_hotspots",
    "proximity": "03_proximity",
    "terrain":   "03b_terrain",
    "risk":      "04_risk",
    "stats":     "05_stats",
}


def module_output_dir(cfg: dict, module_name: str) -> Path:
    """Return (and create) the dedicated output folder for a module.

    Parameters
    ----------
    cfg
        Full config dict; ``cfg["output_dir"]`` is used as the root.
    module_name
        One of "density", "hotspots", "proximity", "terrain", "risk",
        "stats".  Unknown names are used verbatim as the folder name.
    """
    folder_name = _MODULE_FOLDERS.get(module_name, module_name)
    path = Path(cfg["output_dir"]) / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Vector I/O
# -----------------------------------------------------------------------------
def load_vector(path: str, target_crs: str, label: str = "layer") -> gpd.GeoDataFrame:
    """Read a vector file, enforce *target_crs*, drop empty geometries.

    Parameters
    ----------
    path
        Filesystem path to a shapefile / GeoJSON / GPKG.
    target_crs
        CRS string the layer must be in for downstream math (e.g. "EPSG:32630").
    label
        Human-readable name used in log messages.

    Raises
    ------
    ValueError
        If the layer contains zero features after cleaning.
    """
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{label} contains zero features: {path}")

    # Assign CRS if missing, otherwise reproject to target.
    if gdf.crs is None:
        log(f"  WARNING: {label} has no CRS — assuming {target_crs}")
        gdf = gdf.set_crs(target_crs, allow_override=True)
    elif gdf.crs.to_string().upper() != target_crs.upper():
        log(f"  Reprojecting {label}: {gdf.crs} -> {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # Remove null/empty geometries so downstream ops never see them.
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    log(f"  Loaded {label}: {len(gdf)} features | CRS={gdf.crs}")
    return gdf


# -----------------------------------------------------------------------------
# Raster I/O
# -----------------------------------------------------------------------------
def write_geotiff(
    path,
    array: np.ndarray,
    transform,
    crs: str,
    dtype: str = "float32",
    nodata: float = 0.0,
) -> None:
    """Write a single-band GeoTIFF with LZW compression and tiling.

    The profile is kept identical across every raster the pipeline writes so
    that QGIS / ArcGIS display them consistently and downstream tools can
    rely on the same layout.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver":     "GTiff",
        "height":     int(array.shape[0]),
        "width":      int(array.shape[1]),
        "count":      1,
        "dtype":      dtype,
        "crs":        crs,
        "transform":  transform,
        "nodata":     nodata,
        "compress":   "lzw",
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    log(f"  Wrote GeoTIFF -> {path}")


def read_raster_aligned(path: str, height: int, width: int) -> np.ndarray:
    """Read a single-band raster and nearest-neighbour align it to (H, W).

    Used when a source raster was not produced by this pipeline and may have
    a slightly different shape.  Alignment preserves row/column ordering so
    the array stays north-up.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    if arr.shape == (height, width):
        return arr
    rows = np.clip(
        np.linspace(0, arr.shape[0] - 1, height).astype(int),
        0, arr.shape[0] - 1,
    )
    cols = np.clip(
        np.linspace(0, arr.shape[1] - 1, width).astype(int),
        0, arr.shape[1] - 1,
    )
    return arr[np.ix_(rows, cols)]


# -----------------------------------------------------------------------------
# Master grid
# -----------------------------------------------------------------------------
def build_grid_from_bounds(minx, miny, maxx, maxy, pixel: float) -> dict:
    """Build a north-up master grid snapped outward to *pixel*-metre cells.

    Uses ``rasterio.transform.from_bounds`` (not ``from_origin``) so the
    resulting array cannot be vertically flipped by mistake.

    Returns a dict with the following keys:
        minx, miny, maxx, maxy : snapped bounds in metres
        width, height          : grid dimensions in pixels
        pixel                  : pixel size in metres
        transform              : affine transform (row 0 = northern edge)
        xx, yy                 : 2-D arrays of pixel-centre coordinates
    """
    minx = np.floor(minx / pixel) * pixel
    miny = np.floor(miny / pixel) * pixel
    maxx = np.ceil(maxx / pixel) * pixel
    maxy = np.ceil(maxy / pixel) * pixel

    width  = int(round((maxx - minx) / pixel))
    height = int(round((maxy - miny) / pixel))

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    xs = minx + (np.arange(width) + 0.5) * pixel
    ys = maxy - (np.arange(height) + 0.5) * pixel
    xx, yy = np.meshgrid(xs, ys)

    grid = {
        "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
        "width": width, "height": height, "pixel": pixel,
        "transform": transform, "xx": xx, "yy": yy,
    }
    log(f"  Grid: {width} x {height} px ({width * height:,} cells) | "
        f"bounds=({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")
    return grid


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def describe_array(arr: np.ndarray, label: str) -> dict:
    """Log min/max/mean/std of a 2-D array and return the stats as a dict."""
    finite = arr[np.isfinite(arr)]
    stats = {
        "min":  float(np.min(finite))  if finite.size else float("nan"),
        "max":  float(np.max(finite))  if finite.size else float("nan"),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std":  float(np.std(finite))  if finite.size else float("nan"),
    }
    log(f"  [{label}] min={stats['min']:.6g} max={stats['max']:.6g} "
        f"mean={stats['mean']:.6g} std={stats['std']:.6g}")
    return stats

def tier_from_percentiles(arr: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Standardize a continuous array to 1–5 tiers using quintile breaks.

    reverse=False : high values → high tiers (density-like layers)
    reverse=True  : low values  → high tiers (elevation, slope)
    """
    a = np.nan_to_num(arr, nan=0.0)
    breaks = np.percentile(a, [20, 40, 60, 80])
    tiers = np.ones_like(a, dtype=np.float32)
    if reverse:
        tiers[a < breaks[3]] = 2.0
        tiers[a < breaks[2]] = 3.0
        tiers[a < breaks[1]] = 4.0
        tiers[a < breaks[0]] = 5.0
    else:
        tiers[a > breaks[0]] = 2.0
        tiers[a > breaks[1]] = 3.0
        tiers[a > breaks[2]] = 4.0
        tiers[a > breaks[3]] = 5.0
    return tiers


def tier_from_distance(dist_m: np.ndarray,
                       breaks: list[float]) -> np.ndarray:
    """Reclassify a distance raster to 1–5 where close = tier 5.

    *breaks* must be four ascending values, e.g. [500, 1500, 3000, 6000].
    Negative sentinel values (nodata) are treated as far away = tier 1.
    """
    a = np.nan_to_num(dist_m, nan=0.0)
    a[a < 0] = breaks[-1] + 1.0
    tiers = np.ones_like(a, dtype=np.float32)
    tiers[a <= breaks[0]] = 5.0
    tiers[(a > breaks[0]) & (a <= breaks[1])] = 4.0
    tiers[(a > breaks[1]) & (a <= breaks[2])] = 3.0
    tiers[(a > breaks[2]) & (a <= breaks[3])] = 2.0
    return tiers
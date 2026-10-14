"""Getis-Ord Gi* conflict hotspot analysis.

Reads:
    conflicts  (required for this module)

Writes (outputs/02_hotspots/):
    Step2_Conflict_Hotspots_GiStar.shp  (+ .shx, .dbf, .prj, .cpg)
    Conflict_Z_Raster.tif                continuous z-scores
    Conflict_Z_Tier.tif                  reclassified 1–5  (feeds risk.py)
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.features import rasterize

import libpysal
from esda.getisord import G_Local

from .io_utils import (
    log, section, write_geotiff, describe_array, load_vector,
    module_output_dir,
)


def _hex_polygon(cx: float, cy: float, r: float) -> Polygon:
    angles = np.deg2rad(np.arange(6) * 60 + 30)
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    return Polygon(pts)


def _build_hex_grid(study_area, spacing):
    minx, miny, maxx, maxy = study_area.total_bounds
    dx = spacing
    dy = spacing * np.sqrt(3.0) / 2.0
    r = spacing / np.sqrt(3.0)

    n_cols = int(np.ceil((maxx - minx) / dx)) + 3
    n_rows = int(np.ceil((maxy - miny) / dy)) + 3

    geoms = []
    for row in range(n_rows):
        cy = miny + row * dy
        x_shift = dx / 2.0 if (row % 2) else 0.0
        for col in range(n_cols):
            cx = minx + col * dx + x_shift
            geoms.append(_hex_polygon(cx, cy, r))

    hexes = gpd.GeoDataFrame(geometry=geoms, crs=study_area.crs)

    study_union = study_area.geometry.unary_union
    hexes = hexes[hexes.intersects(study_union)].copy()
    hexes["geometry"] = hexes.geometry.intersection(study_union)
    hexes = hexes[~hexes.geometry.is_empty].copy().reset_index(drop=True)
    hexes["hex_id"] = np.arange(len(hexes), dtype=np.int64)

    log(f"  Hex grid: {len(hexes)} cells (spacing {spacing:.0f} m)")
    return hexes


def run(cfg, grid, study_area, active) -> None:
    section("MODULE: hotspots — Getis-Ord Gi*")

    if "conflicts" not in active:
        log("  Skipped: no conflicts layer provided.")
        return

    out_dir = module_output_dir(cfg, "hotspots")
    log(f"  Output folder: {out_dir}")

    conflicts = load_vector(
        active["conflicts"]["path"], cfg["crs"], label="Conflicts"
    )

    spacing = float(cfg.get("hex_spacing", 2000.0))
    hexes = _build_hex_grid(study_area, spacing)

    joined = gpd.sjoin(
        conflicts[["geometry"]],
        hexes[["hex_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    counts = joined.groupby("hex_id").size().rename("NUMPOINTS")

    hexes = hexes.merge(counts, left_on="hex_id", right_index=True, how="left")
    hexes["NUMPOINTS"] = hexes["NUMPOINTS"].fillna(0).astype(int)
    log(f"  NUMPOINTS: min={hexes['NUMPOINTS'].min()} "
        f"max={hexes['NUMPOINTS'].max()} "
        f"total={int(hexes['NUMPOINTS'].sum())}")

    if hexes["NUMPOINTS"].sum() == 0:
        log("  No conflicts fall inside any hexagon — aborting Gi*.")
        return

    centroids = hexes.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])

    w = libpysal.weights.DistanceBand(
        coords, threshold=4000.0, binary=True, silence_warnings=True,
    )
    w.transform = "r"

    islands = [i for i, n in w.neighbors.items() if len(n) == 0]
    if islands:
        log(f"  Dropping {len(islands)} island hex(es)")
        keep = ~np.isin(np.arange(len(hexes)), islands)
        hexes = hexes.loc[keep].reset_index(drop=True)
        coords = coords[keep]
        w = libpysal.weights.DistanceBand(
            coords, threshold=4000.0, binary=True, silence_warnings=True,
        )
        w.transform = "r"

    log(f"  Weights: n={w.n} mean_neighbors={w.mean_neighbors:.2f}")

    y = hexes["NUMPOINTS"].to_numpy(dtype=float)
    gi = G_Local(y, w, star=True, permutations=999, transform="r",
                 seed=42, n_jobs=1)

    hexes["z_score"] = np.asarray(gi.Zs, dtype=float)
    hexes["p_value"] = np.asarray(gi.p_sim, dtype=float)

    hexes["signif"] = "Not Sig."
    hexes.loc[(hexes["p_value"] <= 0.05) & (hexes["z_score"] > 0), "signif"] = "Hot 95%"
    hexes.loc[(hexes["p_value"] <= 0.05) & (hexes["z_score"] < 0), "signif"] = "Cold 95%"
    hexes.loc[(hexes["p_value"] <= 0.01) & (hexes["z_score"] > 0), "signif"] = "Hot 99%"
    hexes.loc[(hexes["p_value"] <= 0.01) & (hexes["z_score"] < 0), "signif"] = "Cold 99%"

    shp_path = out_dir / "Step2_Conflict_Hotspots_GiStar.shp"
    hexes.to_file(shp_path, driver="ESRI Shapefile")
    log(f"  Wrote hotspot layer -> {shp_path}")

    describe_array(hexes["z_score"].to_numpy(), "Z-score across hexagons")

    # Rasterize z-scores onto master grid
    z_raster = rasterize(
        shapes=((geom, float(z)) for geom, z in
                zip(hexes.geometry, hexes["z_score"])),
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        fill=0.0,
        all_touched=False,
        dtype="float32",
    )
    write_geotiff(
        out_dir / "Conflict_Z_Raster.tif",
        z_raster, grid["transform"], cfg["crs"], nodata=0.0,
    )

    # Reclassified tier — the version consumed by risk.py
    tier = np.ones_like(z_raster, dtype=np.float32)
    tier[(z_raster > 0.00) & (z_raster <= 1.00)] = 2.0
    tier[(z_raster > 1.00) & (z_raster <= 1.65)] = 3.0
    tier[(z_raster > 1.65) & (z_raster <= 1.96)] = 4.0
    tier[z_raster > 1.96] = 5.0
    describe_array(tier, "Conflict tier")
    write_geotiff(
        out_dir / "Conflict_Z_Tier.tif",
        tier, grid["transform"], cfg["crs"], dtype="float32", nodata=-9999.0,
    )
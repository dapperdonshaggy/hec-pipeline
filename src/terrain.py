"""Streams, forest, elevation and slope surfaces.

Reads (any subset):
    streams (line)
    forest  (polygon)
    dem     (raster)

Writes (outputs/03b_terrain/):
    Step3_5_Distance_to_Streams.tif           + _Tier.tif
    Step3_5_Distance_to_Forest.tif            + _Tier.tif
    Step3_5_Forest_Edge_Density.tif           + _Tier.tif
    Step3_5_Elevation.tif                     + _Tier.tif
    Step3_5_Slope.tif                         + _Tier.tif
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt, binary_erosion, uniform_filter

from .io_utils import (
    log, section, write_geotiff, describe_array, load_vector,
    module_output_dir, tier_from_percentiles, tier_from_distance,
)


def _distance_from_binary(mask, pixel):
    return (distance_transform_edt(~mask) * pixel).astype(np.float32)


def _reproject_dem_to_master(path, height, width, transform, crs):
    with rasterio.open(path) as src:
        log(f"  DEM: crs={src.crs} shape={src.shape} nodata={src.nodata}")
        dem = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    n_nan = int(np.count_nonzero(~np.isfinite(dem)))
    if n_nan:
        med = float(np.nanmedian(dem))
        log(f"  Filling {n_nan:,} NaN cells with median {med:.2f} m")
        dem = np.where(np.isfinite(dem), dem, med)
    return dem.astype(np.float32)


def _process_streams(active, cfg, grid, out_dir):
    streams = load_vector(active["streams"]["path"], cfg["crs"], label="Streams")

    grid_mask = rasterize(
        shapes=((geom, 1) for geom in streams.geometry),
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        fill=0, all_touched=True, dtype="uint8",
    )

    n = int(grid_mask.sum())
    log(f"  Stream pixels: {n:,}")
    if n == 0:
        log("  No stream pixels on master grid — skipping streams.")
        return

    dist_m = _distance_from_binary(grid_mask.astype(bool), grid["pixel"])
    describe_array(dist_m, "Distance to Streams (m)")
    write_geotiff(
        out_dir / "Step3_5_Distance_to_Streams.tif",
        dist_m, grid["transform"], cfg["crs"], nodata=-9999.0,
    )

    tier = tier_from_distance(dist_m, [1000, 3000, 6000, 9000])
    describe_array(tier, "Streams tier")
    write_geotiff(
        out_dir / "Step3_5_Distance_to_Streams_Tier.tif",
        tier, grid["transform"], cfg["crs"], dtype="float32", nodata=-9999.0,
    )


def _process_forest(active, cfg, grid, out_dir):
    forest = load_vector(active["forest"]["path"], cfg["crs"], label="Forest")

    grid_mask = rasterize(
        shapes=((geom, 1) for geom in forest.geometry),
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        fill=0, all_touched=False, dtype="uint8",
    )

    n = int(grid_mask.sum())
    log(f"  Forest pixels: {n:,} ({100 * n / grid_mask.size:.2f} % of grid)")
    if n == 0:
        log("  No forest pixels on master grid — skipping forest.")
        return

    dist_m = _distance_from_binary(grid_mask.astype(bool), grid["pixel"])
    describe_array(dist_m, "Distance to Forest (m)")
    write_geotiff(
        out_dir / "Step3_5_Distance_to_Forest.tif",
        dist_m, grid["transform"], cfg["crs"], nodata=-9999.0,
    )
    dist_tier = tier_from_distance(dist_m, [500, 2000, 6000, 12000])
    describe_array(dist_tier, "Forest distance tier")
    write_geotiff(
        out_dir / "Step3_5_Distance_to_Forest_Tier.tif",
        dist_tier, grid["transform"], cfg["crs"], dtype="float32", nodata=-9999.0,
    )

    # Edge density
    forest_bool = grid_mask.astype(bool)
    eroded = binary_erosion(forest_bool, structure=np.ones((3, 3)))
    edge = forest_bool & ~eroded

    window_px = int(round(1000.0 / grid["pixel"]))
    window_size = 2 * window_px + 1

    edge_len_per_px = uniform_filter(
        edge.astype(np.float32) * grid["pixel"],
        size=window_size, mode="constant",
    )
    window_area_m2 = (window_size * grid["pixel"]) ** 2
    edge_density = (edge_len_per_px / window_area_m2 * 1e6).astype(np.float32)

    describe_array(edge_density, "Forest Edge Density (km/km²)")
    write_geotiff(
        out_dir / "Step3_5_Forest_Edge_Density.tif",
        edge_density, grid["transform"], cfg["crs"], nodata=-9999.0,
    )
    edge_tier = tier_from_percentiles(edge_density, reverse=False)
    describe_array(edge_tier, "Forest edge tier")
    write_geotiff(
        out_dir / "Step3_5_Forest_Edge_Density_Tier.tif",
        edge_tier, grid["transform"], cfg["crs"], dtype="float32", nodata=-9999.0,
    )


def _process_dem(active, cfg, grid, out_dir):
    dem = _reproject_dem_to_master(
        active["dem"]["path"], grid["height"], grid["width"],
        grid["transform"], cfg["crs"],
    )

    describe_array(dem, "Elevation (m)")
    write_geotiff(
        out_dir / "Step3_5_Elevation.tif",
        dem, grid["transform"], cfg["crs"], nodata=-9999.0,
    )
    # Lower elevation → higher risk
    elev_tier = tier_from_percentiles(dem, reverse=True)
    describe_array(elev_tier, "Elevation tier")
    write_geotiff(
        out_dir / "Step3_5_Elevation_Tier.tif",
        elev_tier, grid["transform"], cfg["crs"], dtype="float32", nodata=-9999.0,
    )

    dy, dx = np.gradient(dem, grid["pixel"], grid["pixel"])
    slope_deg = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)

    describe_array(slope_deg, "Slope (degrees)")
    write_geotiff(
        out_dir / "Step3_5_Slope.tif",
        slope_deg, grid["transform"], cfg["crs"], nodata=-9999.0,
    )
    # Gentler slope → higher risk
    slope_tier = tier_from_percentiles(slope_deg, reverse=True)
    describe_array(slope_tier, "Slope tier")
    write_geotiff(
        out_dir / "Step3_5_Slope_Tier.tif",
        slope_tier, grid["transform"], cfg["crs"], dtype="float32", nodata=-9999.0,
    )


def run(cfg, grid, study_area, active) -> None:
    section("MODULE: terrain — streams, forest, DEM derivatives")

    out_dir = module_output_dir(cfg, "terrain")
    log(f"  Output folder: {out_dir}")

    if "streams" in active:
        _process_streams(active, cfg, grid, out_dir)
    else:
        log("  streams: not provided.")

    if "forest" in active:
        _process_forest(active, cfg, grid, out_dir)
    else:
        log("  forest: not provided.")

    if "dem" in active:
        _process_dem(active, cfg, grid, out_dir)
    else:
        log("  dem: not provided.")
"""Distance-to-farmland and distance-to-settlement surfaces from LULC.

Reads:
    lulc  (required for this module)

Writes (outputs/03_proximity/):
    Step3_Farmland_Mask.tif
    Step3_Settlement_Mask.tif
    Step3_Distance_to_Farms.tif                continuous
    Step3_Distance_to_Farms_Tier.tif           1–5 (feeds risk.py)
    Step3_Distance_to_Settlements.tif          continuous
    Step3_Distance_to_Settlements_Tier.tif     1–5 (feeds risk.py)
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt

from .io_utils import (
    log, section, write_geotiff, describe_array, module_output_dir,
    tier_from_distance,
)


_FARM_BREAKS     = [0, 500, 1500, 3000]
_SETTLE_BREAKS   = [30, 750, 2000, 5000]



def _reproject_to_master(src_path, height, width, transform, crs):
    with rasterio.open(src_path) as src:
        log(f"  Source: shape={src.shape} crs={src.crs} nodata={src.nodata}")
        aligned = np.zeros((height, width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return aligned


def run(cfg, grid, study_area, active) -> None:
    section("MODULE: proximity — distance to farms & settlements")

    if "lulc" not in active:
        log("  Skipped: no LULC raster provided.")
        return

    out_dir = module_output_dir(cfg, "proximity")
    log(f"  Output folder: {out_dir}")

    spec = active["lulc"]
    farm_code = int(spec.get("farm_code", 4))
    settle_code = int(spec.get("settlement_code", 5))

    height, width = grid["height"], grid["width"]
    transform = grid["transform"]
    pixel = grid["pixel"]

    log(f"  Using farmland code = {farm_code}, "
        f"settlement code = {settle_code}")

    lulc = _reproject_to_master(spec["path"], height, width, transform, cfg["crs"])

    vals, counts = np.unique(lulc, return_counts=True)
    log("  Class histogram after alignment:")
    for v, c in zip(vals, counts):
        log(f"      code {int(v):>6}  ->  {int(c):>12,} px")

    farm_mask = (lulc == farm_code)
    settle_mask = (lulc == settle_code)

    n_farm = int(farm_mask.sum())
    n_settle = int(settle_mask.sum())

    if n_farm == 0:
        raise ValueError(
            f"Zero farmland pixels for code {farm_code}. "
            f"Codes present: {vals.tolist()}"
        )
    if n_settle == 0:
        raise ValueError(
            f"Zero settlement pixels for code {settle_code}. "
            f"Codes present: {vals.tolist()}"
        )

    n_valid = int(np.count_nonzero(lulc))
    log(f"  Farmland   : {n_farm:,} px "
        f"({100 * n_farm / n_valid:.2f} % of valid)")
    log(f"  Settlement : {n_settle:,} px "
        f"({100 * n_settle / n_valid:.2f} % of valid)")

    # Masks
    write_geotiff(
        out_dir / "Step3_Farmland_Mask.tif",
        farm_mask.astype(np.uint8), transform, cfg["crs"],
        dtype="uint8", nodata=255,
    )
    write_geotiff(
        out_dir / "Step3_Settlement_Mask.tif",
        settle_mask.astype(np.uint8), transform, cfg["crs"],
        dtype="uint8", nodata=255,
    )

    # Continuous distance surfaces
    log("  Computing Euclidean distance transforms ...")
    farm_dist_m   = (distance_transform_edt(~farm_mask)   * pixel).astype(np.float32)
    settle_dist_m = (distance_transform_edt(~settle_mask) * pixel).astype(np.float32)

    describe_array(farm_dist_m,   "Distance to Farms (m)")
    describe_array(settle_dist_m, "Distance to Settlements (m)")

    write_geotiff(
        out_dir / "Step3_Distance_to_Farms.tif",
        farm_dist_m, transform, cfg["crs"], nodata=-9999.0,
    )
    write_geotiff(
        out_dir / "Step3_Distance_to_Settlements.tif",
        settle_dist_m, transform, cfg["crs"], nodata=-9999.0,
    )

    # Reclassified tiers — consumed by risk.py
    farm_tier   = tier_from_distance(farm_dist_m,   _FARM_BREAKS)
    settle_tier = tier_from_distance(settle_dist_m, _SETTLE_BREAKS)

    describe_array(farm_tier,   "Farms tier")
    describe_array(settle_tier, "Settlements tier")

    write_geotiff(
        out_dir / "Step3_Distance_to_Farms_Tier.tif",
        farm_tier, transform, cfg["crs"], dtype="float32", nodata=-9999.0,
    )
    write_geotiff(
        out_dir / "Step3_Distance_to_Settlements_Tier.tif",
        settle_tier, transform, cfg["crs"], dtype="float32", nodata=-9999.0,
    )
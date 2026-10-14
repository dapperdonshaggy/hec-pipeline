"""Weighted overlay of pre-reclassified tier rasters.

Reads the *_Tier.tif files produced by each module.  No reclassification
happens here — what each module wrote is what contributes to the final map.

Writes (outputs/04_risk/):
    Step4_Final_Community_Risk_Map.tif
"""
from __future__ import annotations

import numpy as np
from rasterio.features import geometry_mask

from .io_utils import (
    log, section, write_geotiff, describe_array, read_raster_aligned,
    module_output_dir,
)


def _load_tier(path, height, width) -> np.ndarray:
    """Read a tier raster, clean sentinels, and clip values to 1–5."""
    arr = read_raster_aligned(str(path), height, width)
    arr = np.nan_to_num(arr, nan=0.0)
    arr[arr < 1] = 1.0
    arr[arr > 5] = 5.0
    return arr


def run(cfg, grid, study_area, active) -> None:
    section("MODULE: risk — weighted overlay")

    height, width = grid["height"], grid["width"]
    transform = grid["transform"]

    out_dir = module_output_dir(cfg, "risk")
    density_dir   = module_output_dir(cfg, "density")
    hotspots_dir  = module_output_dir(cfg, "hotspots")
    proximity_dir = module_output_dir(cfg, "proximity")
    terrain_dir   = module_output_dir(cfg, "terrain")
    log(f"  Output folder: {out_dir}")

    contributions: list[tuple[str, np.ndarray, float]] = []

    def _add(name: str, tiers: np.ndarray, weight: float) -> None:
        contributions.append((name, tiers, float(weight)))
        uniq, counts = np.unique(tiers, return_counts=True)
        dist = " | ".join(f"T{int(u)}={int(c):,}" for u, c in zip(uniq, counts))
        log(f"  Added {name:<12} (weight {weight:.3f})  tiers: {dist}")

    # -------------------------------------------------------------------------
    # Elephant
    # -------------------------------------------------------------------------
    if "sightings" in active or "pathways" in active:
        p = density_dir / "Step1C_Combined_Elephant_Tier.tif"
        if p.exists():
            tiers = _load_tier(p, height, width)
            _add("Elephant", tiers,
                 cfg.get("elephant_presence_weight", 0.25))

    # -------------------------------------------------------------------------
    # Conflict
    # -------------------------------------------------------------------------
    if "conflicts" in active:
        p = hotspots_dir / "Conflict_Z_Tier.tif"
        if p.exists():
            tiers = _load_tier(p, height, width)
            _add("Conflict", tiers, active["conflicts"]["weight"])

    # -------------------------------------------------------------------------
    # Farms & Settlements
    # -------------------------------------------------------------------------
    if "lulc" in active:
        spec = active["lulc"]
        if "weight_farms" in spec:
            p = proximity_dir / "Step3_Distance_to_Farms_Tier.tif"
            if p.exists():
                tiers = _load_tier(p, height, width)
                _add("Farms", tiers, spec["weight_farms"])
        if "weight_settlements" in spec:
            p = proximity_dir / "Step3_Distance_to_Settlements_Tier.tif"
            if p.exists():
                tiers = _load_tier(p, height, width)
                _add("Settlements", tiers, spec["weight_settlements"])

    # -------------------------------------------------------------------------
    # Streams
    # -------------------------------------------------------------------------
    if "streams" in active:
        p = terrain_dir / "Step3_5_Distance_to_Streams_Tier.tif"
        if p.exists():
            tiers = _load_tier(p, height, width)
            _add("Streams", tiers, active["streams"]["weight"])

    # -------------------------------------------------------------------------
    # Forest
    # -------------------------------------------------------------------------
    if "forest" in active:
        spec = active["forest"]
        if "weight_distance" in spec:
            p = terrain_dir / "Step3_5_Distance_to_Forest_Tier.tif"
            if p.exists():
                tiers = _load_tier(p, height, width)
                _add("ForestDist", tiers, spec["weight_distance"])
        if "weight_edge" in spec:
            p = terrain_dir / "Step3_5_Forest_Edge_Density_Tier.tif"
            if p.exists():
                tiers = _load_tier(p, height, width)
                _add("ForestEdge", tiers, spec["weight_edge"])

    # -------------------------------------------------------------------------
    # DEM
    # -------------------------------------------------------------------------
    if "dem" in active:
        spec = active["dem"]
        if "weight_elevation" in spec:
            p = terrain_dir / "Step3_5_Elevation_Tier.tif"
            if p.exists():
                tiers = _load_tier(p, height, width)
                _add("Elevation", tiers, spec["weight_elevation"])
        if "weight_slope" in spec:
            p = terrain_dir / "Step3_5_Slope_Tier.tif"
            if p.exists():
                tiers = _load_tier(p, height, width)
                _add("Slope", tiers, spec["weight_slope"])

    # -------------------------------------------------------------------------
    # Weighted sum
    # -------------------------------------------------------------------------
    if not contributions:
        log("  No tier rasters available — risk module skipped.")
        return

    total_w = sum(w for _, _, w in contributions)
    log(f"  Total weight = {total_w:.4f}")
    if abs(total_w - 1.0) > 1e-6:
        log("  Rescaling weights to sum to 1.0")
        contributions = [(n, t, w / total_w) for n, t, w in contributions]

    risk = np.zeros((height, width), dtype=np.float32)
    for name, tiers, w in contributions:
        risk += tiers.astype(np.float32) * w
        log(f"    {name:<12} × {w:.4f}")

    describe_array(risk, "Risk (pre-mask)")

    # -------------------------------------------------------------------------
    # Clip to study area
    # -------------------------------------------------------------------------
    log("  Masking to study-area boundary ...")
    inside = geometry_mask(
        geometries=[study_area.geometry.union_all()],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )
    risk_masked = np.where(inside, risk, 0.0).astype(np.float32)

    valid = risk_masked[risk_masked > 0]
    if valid.size:
        log(f"  Final risk: min={valid.min():.4f} max={valid.max():.4f} "
            f"mean={valid.mean():.4f} cells={valid.size:,}")

    write_geotiff(
        out_dir / "Step4_Final_Community_Risk_Map.tif",
        risk_masked, transform, cfg["crs"], nodata=0.0,
    )
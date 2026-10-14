"""Point kernel density estimation and line density surfaces.

Reads:
    sightings  (optional, point)
    pathways   (optional, line)

Writes (outputs/01_density/):
    Step1A_Point_Density_Only.tif          continuous
    Step1B_Line_Density_Only.tif           continuous
    Step1C_Combined_Elephant_Master.tif    continuous
    Step1C_Combined_Elephant_Tier.tif      reclassified 1–5 (feeds risk.py)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from rasterio.features import rasterize

from .io_utils import (
    log, section, write_geotiff, describe_array, load_vector,
    module_output_dir, tier_from_percentiles,
)


_NEG_EPS = 1e-10


def _build_kernel(search_radius: float, pixel: float) -> tuple[np.ndarray, float]:
    """Build a quartic (biweight) kernel and return (kernel, circle_area_m2).

    The kernel falls smoothly to zero at the search radius, so overlapping
    neighbourhoods blend without producing visible disc edges.

    The geometric area of the search circle is returned separately so the
    caller can divide the convolution result to obtain a true density.
    """
    radius_px = int(np.ceil(search_radius / pixel))
    ksize = 2 * radius_px + 1

    ky, kx = np.mgrid[0:ksize, 0:ksize]
    dist_px = np.sqrt((kx - radius_px) ** 2 + (ky - radius_px) ** 2)
    u = dist_px / radius_px

    # Quartic (biweight) kernel: 3/π · (1 − u²)²  for u ≤ 1
    kernel = np.where(u <= 1.0, (3.0 / np.pi) * (1.0 - u ** 2) ** 2, 0.0)

    circle_area_m2 = float(np.pi * search_radius ** 2)

    log(f"  Kernel footprint: {ksize}x{ksize} px "
        f"(quartic/biweight) | "
        f"circle area = {circle_area_m2:,.2f} m²")
    return kernel, circle_area_m2


def _convolve_and_clean(grid_values, kernel, area_m2, label):
    """Convolve, divide by area, clamp FFT round-trip noise."""
    convolved = fftconvolve(grid_values, kernel, mode="same")
    density = (convolved / area_m2).astype(np.float32)

    n_neg = int(np.count_nonzero(density < 0))
    if n_neg:
        log(f"  {label}: clamping {n_neg} negative FFT-noise cells to 0")
    density[density < _NEG_EPS] = 0.0
    return density


def run(cfg, grid, study_area, active) -> None:
    """Compute point KDE and line density on the master grid."""
    section("MODULE: density — Point KDE + Line Density")

    transform = grid["transform"]
    height, width = grid["height"], grid["width"]
    pixel = grid["pixel"]
    search_radius = float(cfg["search_radius"])
    out_dir = module_output_dir(cfg, "density")
    log(f"  Output folder: {out_dir}")

    kernel, area_m2 = _build_kernel(search_radius, pixel)

    # -------------------------------------------------------------------------
    # Phase 1A — Point density
    # -------------------------------------------------------------------------
    point_density = None
    if "sightings" in active:
        log("Phase 1A — rasterising sighting points ...")
        sightings = load_vector(
            active["sightings"]["path"], cfg["crs"], label="Sightings"
        )

        if "herd_est" in sightings.columns:
            weights = (
                sightings["herd_est"]
                .apply(lambda v: float(v) if v is not None else 0.0)
                .clip(lower=0.0)
                .to_numpy()
            )
            if weights.sum() <= 0:
                log("  'herd_est' present but zero — using weight 1.0")
                weights = np.ones(len(sightings))
            else:
                log(f"  Using 'herd_est' as point weight "
                    f"(sum={weights.sum():.1f})")
        else:
            weights = np.ones(len(sightings))
            log("  No 'herd_est' column — weight 1.0 per point")

        point_grid = rasterize(
            shapes=((geom, float(w))
                    for geom, w in zip(sightings.geometry, weights)),
            out_shape=(height, width),
            transform=transform,
            fill=0.0,
            all_touched=False,
            dtype="float32",
        )
        describe_array(point_grid, "Phase 1A raw point grid")

        point_density = _convolve_and_clean(
            point_grid, kernel, area_m2, "Phase 1A"
        )
        describe_array(point_density, "Phase 1A point density (pts/m²)")
        write_geotiff(
            out_dir / "Step1A_Point_Density_Only.tif",
            point_density, transform, cfg["crs"], nodata=-9999.0,
        )
    else:
        log("  Skipped: no sightings layer provided.")

    # -------------------------------------------------------------------------
    # Phase 1B — Line density
    # -------------------------------------------------------------------------
    line_density = None
    if "pathways" in active:
        log("Phase 1B — rasterising migration pathways ...")
        pathways = load_vector(
            active["pathways"]["path"], cfg["crs"], label="Pathways"
        )

        line_grid = rasterize(
            shapes=((geom, 1.0) for geom in pathways.geometry),
            out_shape=(height, width),
            transform=transform,
            fill=0.0,
            all_touched=True,
            dtype="float32",
        )
        describe_array(line_grid, "Phase 1B raw line presence grid")

        convolved = fftconvolve(line_grid, kernel, mode="same")
        line_density = ((convolved * pixel) / area_m2).astype(np.float32)

        n_neg = int(np.count_nonzero(line_density < 0))
        if n_neg:
            log(f"  Phase 1B: clamping {n_neg} negative FFT-noise cells")
        line_density[line_density < _NEG_EPS] = 0.0

        describe_array(line_density, "Phase 1B line density (m/m²)")
        write_geotiff(
            out_dir / "Step1B_Line_Density_Only.tif",
            line_density, transform, cfg["crs"], nodata=-9999.0,
        )
    else:
        log("  Skipped: no pathways layer provided.")

    # -------------------------------------------------------------------------
    # Phase 1C — Combined master + reclassified tier
    # -------------------------------------------------------------------------
    if point_density is None and line_density is None:
        log("  Neither point nor line density available — nothing to combine.")
        return

    if point_density is None:
        point_density = np.zeros((height, width), dtype=np.float32)
    if line_density is None:
        line_density = np.zeros((height, width), dtype=np.float32)

    log("Phase 1C — combining point + line densities ...")
    combined = (point_density + line_density).astype(np.float32)
    combined[combined < _NEG_EPS] = 0.0
    describe_array(combined, "Phase 1C combined master")

    write_geotiff(
        out_dir / "Step1C_Combined_Elephant_Master.tif",
        combined, transform, cfg["crs"], nodata=-9999.0,
    )

    # Reclassified tier — the version consumed by risk.py
    tier = tier_from_percentiles(combined, reverse=False)
    describe_array(tier, "Phase 1C elephant tier")
    write_geotiff(
        out_dir / "Step1C_Combined_Elephant_Tier.tif",
        tier, transform, cfg["crs"], dtype="float32", nodata=-9999.0,
    )
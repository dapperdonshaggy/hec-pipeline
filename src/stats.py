"""Final descriptive statistics report for the risk map.

Reads:
    outputs/Step4_Final_Community_Risk_Map.tif
    (and optionally every intermediate raster for correlation diagnostics)

Writes:
    Nothing on disk.  Pure reporting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .io_utils import log, section


def _bin_histogram(valid: np.ndarray,
                   edges: np.ndarray) -> None:
    """Print a text histogram over the given bin edges."""
    hist, _ = np.histogram(valid, bins=edges)
    total = valid.size
    for i, h in enumerate(hist):
        lo, hi = edges[i], edges[i + 1]
        pct = 100.0 * h / total
        bar = "█" * int(round(pct / 2))
        log(f"  [{lo:.1f} – {hi:.1f})  {h:>10,} px  {pct:>6.2f} %  {bar}")


def run(cfg, grid, study_area, active) -> None:
    """Print a comprehensive statistics report for the final risk map."""
    section("MODULE: stats — final statistics report")

    out_dir = Path(cfg["output_dir"])
    risk_path = out_dir / "Step4_Final_Community_Risk_Map.tif"
    if not risk_path.exists():
        log("  Risk raster not found — nothing to report.")
        return

    with rasterio.open(risk_path) as src:
        risk = src.read(1, masked=True).astype(np.float32)
        transform = src.transform
        pixel = transform.a

    arr = risk.filled(np.nan)
    valid_mask = np.isfinite(arr) & (arr > 0)
    valid = arr[valid_mask]

    n_total = arr.size
    n_valid = int(valid.size)
    log(f"  Grid          : {arr.shape[1]} x {arr.shape[0]} px @ {pixel:.2f} m")
    log(f"  Total cells   : {n_total:,}")
    log(f"  Valid cells   : {n_valid:,}  ({100 * n_valid / n_total:.2f} %)")
    log(f"  Background    : {n_total - n_valid:,}")

    if n_valid == 0:
        log("  No valid cells — report aborted.")
        return

    # -------------------------------------------------------------------------
    # Global descriptive statistics
    # -------------------------------------------------------------------------
    log("")
    log("-" * 78)
    log("GLOBAL DESCRIPTIVE STATISTICS (valid cells only)")
    log("-" * 78)
    stats = {
        "Count":    n_valid,
        "Minimum":  float(valid.min()),
        "Maximum":  float(valid.max()),
        "Mean":     float(valid.mean()),
        "Median":   float(np.median(valid)),
        "Std Dev":  float(valid.std()),
        "Variance": float(valid.var()),
        "Range":    float(valid.max() - valid.min()),
        "P05":      float(np.percentile(valid, 5)),
        "P25":      float(np.percentile(valid, 25)),
        "P50":      float(np.percentile(valid, 50)),
        "P75":      float(np.percentile(valid, 75)),
        "P95":      float(np.percentile(valid, 95)),
        "P99":      float(np.percentile(valid, 99)),
        "Skewness": float(((valid - valid.mean()) ** 3).mean() / valid.std() ** 3),
        "Kurtosis": float(((valid - valid.mean()) ** 4).mean() / valid.std() ** 4 - 3),
    }
    for k, v in stats.items():
        if k == "Count":
            log(f"  {k:<12}: {int(v):,}")
        else:
            log(f"  {k:<12}: {v:.6f}")

    # -------------------------------------------------------------------------
    # Tier histogram
    # -------------------------------------------------------------------------
    log("")
    log("-" * 78)
    log("TIER DISTRIBUTION (0.5-width bins)")
    log("-" * 78)
    _bin_histogram(valid, np.arange(1.0, 5.51, 0.5))

    # -------------------------------------------------------------------------
    # Integer class summary with area
    # -------------------------------------------------------------------------
    log("")
    log("-" * 78)
    log("INTEGER RISK CLASS SUMMARY")
    log("-" * 78)
    risk_int = np.clip(np.round(valid), 1, 5).astype(int)
    pixel_ha  = (pixel * pixel) / 10_000.0
    pixel_km2 = pixel_ha / 100.0

    labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
    log(f"  {'Class':<6} {'Label':<12} {'Cells':>12} {'%':>8} "
        f"{'Area (ha)':>14} {'Area (km²)':>14}")
    for c in range(1, 6):
        n_c = int((risk_int == c).sum())
        pct = 100.0 * n_c / n_valid
        log(f"  {c:<6} {labels[c]:<12} {n_c:>12,} {pct:>7.2f}% "
            f"{n_c * pixel_ha:>14,.1f} {n_c * pixel_km2:>14,.2f}")

    # -------------------------------------------------------------------------
    # Threshold analysis
    # -------------------------------------------------------------------------
    log("")
    log("-" * 78)
    log("HOTSPOT THRESHOLD ANALYSIS")
    log("-" * 78)
    for t in (3.0, 3.5, 4.0, 4.5):
        n_above = int((valid >= t).sum())
        log(f"  Risk ≥ {t:.1f}  :  {n_above:>10,} cells  "
            f"({100 * n_above / n_valid:>6.2f} %)  =  "
            f"{n_above * pixel_ha:>12,.1f} ha")

    # -------------------------------------------------------------------------
    # Layer correlation diagnostics
    # -------------------------------------------------------------------------
    log("")
    log("-" * 78)
    log("LAYER CORRELATION VS FINAL RISK (Pearson r)")
    log("-" * 78)

    layer_files = {
        "Elephant":     "Step1C_Combined_Elephant_Master.tif",
        "Conflict Z":   "Conflict_Z_Raster.tif",
        "Dist Farms":   "Step3_Distance_to_Farms.tif",
        "Dist Settle":  "Step3_Distance_to_Settlements.tif",
        "Dist Streams": "Step3_5_Distance_to_Streams.tif",
        "Dist Forest":  "Step3_5_Distance_to_Forest.tif",
        "Forest Edge":  "Step3_5_Forest_Edge_Density.tif",
        "Elevation":    "Step3_5_Elevation.tif",
        "Slope":        "Step3_5_Slope.tif",
    }
    risk_flat = arr[valid_mask]
    for label, fname in layer_files.items():
        p = out_dir / fname
        if not p.exists():
            log(f"  {label:<14} (missing)")
            continue
        with rasterio.open(p) as s:
            other = s.read(1).astype(np.float32)
        if other.shape != arr.shape:
            continue
        other_flat = other[valid_mask]
        other_flat = np.nan_to_num(other_flat, nan=0.0, posinf=0.0, neginf=0.0)
        if other_flat.std() == 0 or risk_flat.std() == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(other_flat, risk_flat)[0, 1])
        log(f"  {label:<14} r = {r:+.4f}")

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("FINAL VERDICT")
    log("=" * 78)
    log(f"  Study-area extent        : {n_valid * pixel_km2:,.2f} km²")
    log(f"  Mean risk score          : {stats['Mean']:.4f}")
    log(f"  Median risk score        : {stats['Median']:.4f}")
    log(f"  High + Very High (≥4.0)  : "
        f"{100 * (valid >= 4.0).sum() / n_valid:.2f} % of study area")
    log("=" * 78)
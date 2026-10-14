"""HEC Pipeline — master sequential runner.

Runs each module in order, skipping those whose required inputs are not
available.  Weights are renormalized once before the risk module so the
final map is always on the same 1–5 scale.

Usage
-----
    python run_all.py                    # full pipeline
    python run_all.py --config my.yaml   # custom config file
    python run_all.py --only density risk
    python run_all.py --stop-on-error
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

# Ensure `src` is importable when executed directly.
sys.path.insert(0, str(Path(__file__).parent))

from src.io_utils import (
    log, section, layer_exists, load_vector, build_grid_from_bounds,
)
from src import density, hotspots, proximity, terrain, risk


# -----------------------------------------------------------------------------
# Module registry — the single source of truth for execution order and
# per-module input requirements.
# -----------------------------------------------------------------------------
MODULES = [
    {"name": "density",   "module": density,
     "requires": ["sightings", "pathways"], "mode": "any"},
    {"name": "hotspots",  "module": hotspots,
     "requires": ["conflicts"],             "mode": "all"},
    {"name": "proximity", "module": proximity,
     "requires": ["lulc"],                  "mode": "all"},
    {"name": "terrain",   "module": terrain,
     "requires": ["streams", "forest", "dem"], "mode": "any"},
    {"name": "risk",      "module": risk,
     "requires": [],                        "mode": "all"},
]


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Read the YAML config file and return it as a dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    log(f"Loaded config: {path}")
    return cfg


def scan_layers(cfg: dict) -> dict:
    """Return {name: spec} for layers whose files exist on disk."""
    active = {}
    for name, spec in cfg.get("layers", {}).items():
        if layer_exists(spec.get("path")):
            active[name] = spec
            log(f"  ACTIVE  : {name}  ({spec['path']})")
        else:
            log(f"  MISSING : {name}  ({spec.get('path')})")
    return active


def can_run(mod: dict, active: dict) -> tuple[bool, str]:
    """Decide whether *mod* can run given the active layers."""
    reqs = mod["requires"]
    if not reqs:
        return True, "always runs"
    present = [r for r in reqs if r in active]
    if mod["mode"] == "all":
        ok = len(present) == len(reqs)
        reason = "ok" if ok else f"missing {sorted(set(reqs) - set(present))}"
    else:
        ok = len(present) > 0
        reason = f"has {present}" if ok else "none present"
    return ok, reason


def renormalize_weights(active: dict) -> dict:
    """Rescale every positive weight so the set sums to 1.0."""
    total = 0.0
    for spec in active.values():
        for k, v in spec.items():
            if k.startswith("weight") and isinstance(v, (int, float)) and v > 0:
                total += v
    if total == 0:
        return active
    log(f"Renormalizing weights: raw sum = {total:.4f}")
    for spec in active.values():
        for k, v in list(spec.items()):
            if k.startswith("weight") and isinstance(v, (int, float)) and v > 0:
                spec[k] = v / total
    return active


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the HEC spatial analysis pipeline."
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to the YAML config (default: config.yaml)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only the named modules")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Halt on the first module failure")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Validate required top-level keys before doing anything else.
    required_keys = [
        "crs", "pixel_size", "search_radius",
        "input_dir", "output_dir", "study_area", "layers",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        log(f"FATAL: config '{args.config}' missing keys: {missing}")
        log(f"       keys present: {sorted(cfg.keys())}")
        sys.exit(1)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["input_dir"]).mkdir(parents=True, exist_ok=True)

    section("HEC PIPELINE — run_all")

    # -------------------------------------------------------------------------
    # Study area (required)
    # -------------------------------------------------------------------------
    sa_path = cfg["study_area"].get("path")
    if not layer_exists(sa_path):
        log(f"FATAL: study area not found at {sa_path}")
        sys.exit(1)
    study_area = load_vector(sa_path, cfg["crs"], label="Study Area")

    # -------------------------------------------------------------------------
    # Master grid
    # -------------------------------------------------------------------------
    grid = build_grid_from_bounds(
        *study_area.total_bounds, pixel=float(cfg["pixel_size"])
    )
    meta_path = Path(cfg["output_dir"]) / "grid_metadata.npz"
    np.savez(
        meta_path,
        minx=grid["minx"], miny=grid["miny"],
        maxx=grid["maxx"], maxy=grid["maxy"],
        width=grid["width"], height=grid["height"],
        pixel=grid["pixel"],
    )
    log(f"  Grid metadata cached -> {meta_path}")

    # -------------------------------------------------------------------------
    # Input scan
    # -------------------------------------------------------------------------
    log("Scanning inputs ...")
    active = scan_layers(cfg)

    # -------------------------------------------------------------------------
    # Weight renormalization (once, before running any module)
    # -------------------------------------------------------------------------
    active = renormalize_weights(active)

    # -------------------------------------------------------------------------
    # Optional subset filter
    # -------------------------------------------------------------------------
    selected = MODULES
    if args.only:
        selected = [m for m in MODULES if m["name"] in args.only]
        log(f"Running subset: {[m['name'] for m in selected]}")

    # -------------------------------------------------------------------------
    # Execute modules
    # -------------------------------------------------------------------------
    results = []
    t_start = time.time()

    for mod in selected:
        name = mod["name"]
        ok, reason = can_run(mod, active)

        log("")
        log("-" * 78)
        log(f"MODULE: {name}  [{reason}]")
        log("-" * 78)

        if not ok:
            results.append((name, "SKIPPED", 0.0, reason))
            continue

        t0 = time.time()
        try:
            mod["module"].run(cfg, grid, study_area, active)
            dur = time.time() - t0
            results.append((name, "OK", dur, reason))
            log(f"  ✓ {name} completed in {dur:.1f} s")

        except Exception as exc:
            dur = time.time() - t0
            results.append((name, "FAILED", dur, str(exc)))
            log(f"  ✗ {name} FAILED after {dur:.1f} s: {exc}")
            traceback.print_exc()
            if args.stop_on_error:
                log("Halting (--stop-on-error).")
                break

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - t_start
    log("")
    section("PIPELINE SUMMARY")
    log(f"  {'MODULE':<12} {'STATUS':<10} {'DURATION':>10}  REASON")
    log("  " + "-" * 74)
    for name, status, dur, reason in results:
        dur_str = f"{dur:.1f} s" if dur else "—"
        log(f"  {name:<12} {status:<10} {dur_str:>10}  {reason}")
    log("  " + "-" * 74)
    n_ok = sum(1 for r in results if r[1] == "OK")
    n_skip = sum(1 for r in results if r[1] == "SKIPPED")
    n_fail = sum(1 for r in results if r[1] == "FAILED")
    log(f"  Total time : {total_time:.1f} s")
    log(f"  Ran / Skipped / Failed : {n_ok} / {n_skip} / {n_fail}")
    log("=" * 78)

    if n_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
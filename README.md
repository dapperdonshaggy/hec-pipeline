# HEC Pipeline

A modular Python pipeline for human–elephant conflict (HEC) spatial analysis.
Processes vector and raster inputs into a weighted community risk map plus
supporting intermediate layers.

---

## Quick start

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate hec

# 2. Drop your source files into inputs/
ls inputs/

# 3. Review and edit config.yaml if your filenames differ
#    (the default paths match the project's original naming)

# 4. Run the pipeline
python run_all.py
```

All outputs are written to `outputs/`. Nothing else is required.

---

## Design principles

1. **Every layer is optional.** The pipeline uses whatever is present in
   `inputs/` and skips the rest. Weights are renormalized automatically
   so the final risk map is always on the same 1–5 scale.
2. **Master grid is built once.** The study-area extent defines a single
   30 m grid that every module shares, so all outputs are pixel-aligned
   by construction.
3. **CRS is fixed at EPSG:32630.** Every layer is reprojected on read,
   so input CRS variations never leak into downstream math.
4. **Modules are independent.** Each module is callable in isolation and
   writes its own outputs; running a subset with `--only` is safe.

---

## Project structure

```
hec-pipeline/
├── config.yaml            # user-editable paths and parameters
├── environment.yml        # conda environment
├── run.py                 # thin wrapper -> run_all.main
├── run_all.py             # master sequential runner
├── src/
│   ├── __init__.py
│   ├── io_utils.py        # shared I/O, logging, grid helpers
│   ├── density.py         # KDE + line density
│   ├── hotspots.py        # Getis-Ord Gi*
│   ├── proximity.py       # distance to farms / settlements
│   ├── terrain.py         # streams, forest, DEM derivatives
│   ├── risk.py            # weighted overlay
│   └── stats.py           # optional statistics report
├── inputs/                # drop source files here
├── outputs/               # pipeline writes results here
└── docs/                  # notes and methods write-ups
```

---

## Modules

| Module          | Requires                  | Produces                                                       |
|-----------------|---------------------------|----------------------------------------------------------------|
| `density.py`    | sightings and/or pathways | `Step1A_Point_Density_Only.tif`, `Step1B_Line_Density_Only.tif`, `Step1C_Combined_Elephant_Master.tif` |
| `hotspots.py`   | conflicts                 | `Step2_Conflict_Hotspots_GiStar.shp`, `Conflict_Z_Raster.tif` |
| `proximity.py`  | lulc                      | `Step3_Farmland_Mask.tif`, `Step3_Settlement_Mask.tif`, `Step3_Distance_to_Farms.tif`, `Step3_Distance_to_Settlements.tif` |
| `terrain.py`    | streams, forest, dem      | `Step3_5_Distance_to_Streams.tif`, `Step3_5_Distance_to_Forest.tif`, `Step3_5_Forest_Edge_Density.tif`, `Step3_5_Elevation.tif`, `Step3_5_Slope.tif` |
| `risk.py`       | any of the above          | `Step4_Final_Community_Risk_Map.tif`                          |
| `stats.py`      | Step 4 output             | (console report only — no file output)                        |

---

## Runners

```bash
# Full pipeline
python run_all.py

# Custom config
python run_all.py --config my_config.yaml

# Only specific modules
python run_all.py --only density risk

# Halt on first failure (useful in CI)
python run_all.py --stop-on-error
```

Exit codes:

| Code | Meaning                                   |
|------|-------------------------------------------|
| 0    | Success                                   |
| 1    | Study area missing or unrecoverable error |
| 2    | One or more modules failed                |
| 3    | Unknown module name passed to `--only`    |

---

## Configuration

`config.yaml` is the only file a user edits. Key parameters:

| Key             | Default          | Meaning                                    |
|-----------------|------------------|--------------------------------------------|
| `crs`           | `EPSG:32630`     | Target CRS for all outputs                 |
| `pixel_size`    | `30.0`           | Master grid resolution in metres           |
| `search_radius` | `5000.0`         | Bandwidth for KDE and line density         |
| `hex_spacing`   | `2000.0`         | Hexagon size for the Gi* aggregation grid  |
| `input_dir`     | `inputs`         | Where source files are read from           |
| `output_dir`    | `outputs`        | Where results are written                  |

Every layer under `layers:` may be omitted. If a file is missing, its module
is skipped and its weight is redistributed across the remaining layers.

---

## Outputs

All outputs land in `outputs/` on the master grid. They are written as
single-band LZW-compressed GeoTIFFs (float32, nodata = −9999 for continuous
surfaces; uint8 nodata = 255 for binary masks), plus one shapefile for the
Gi* hotspot layer.

**Data notes:**

- `Step4_Final_Community_Risk_Map.tif` is the primary deliverable.
  Values range from 1.0 (lowest risk) to 5.0 (highest risk).
  Background outside the study-area boundary is set to 0 (nodata).
- Distance rasters (`Step3_*`, `Step3_5_*`) have real values starting at
  0 inside the source feature. Nodata is −9999 so `0` remains a valid
  measurement.
- `Conflict_Z_Raster.tif` is the rasterized Gi* z-score surface, used
  by `risk.py`. The authoritative per-hexagon statistics live in the
  shapefile.

---

## Validation workflow

After a run, the recommended checks in QGIS or ArcGIS:

1. Open `Step4_Final_Community_Risk_Map.tif` and style it as a 1–5
   classified ramp. The extent should match the study-area polygon
   exactly, with no rectangle overhang.
2. Overlay the study-area boundary. Background pixels should be
   transparent (nodata).
3. Open `Step3_Distance_to_Farms.tif`. The zero-value edge should trace
   the farmland polygons from the LULC. If it is offset, the LULC and
   master grid are misaligned — check that the LULC was reprojected
   onto the same extent.
4. Open `Step2_Conflict_Hotspots_GiStar.shp`. Hotspots (red) should
   correspond to visible clusters of historic conflict points.

---

## Common issues

**All outputs are blank.** The study-area shapefile likely does not
overlap any of the input layers. Check `study_area.total_bounds`
against each layer's bounds in the module logs.

**Farmland or settlement masks are empty.** The `farm_code` /
`settlement_code` in `config.yaml` do not match the LULC legend.
Run `python -m src.proximity` in isolation and inspect the class
histogram printed to the log.

**Gi* z-scores are all zero.** Either no conflicts fall inside any
hexagon (check that `hex_spacing` is reasonable for the study area),
or the shapefile has no point features. The module logs both the
hexagon count and the total point count.

**Distances start at 30 m instead of 0.** This is correct behaviour
when `nodata = 0`. Farm interiors hold a value of 0, but QGIS hides
them because 0 is declared nodata. The distance rasters in this
pipeline use `nodata = -9999` precisely to avoid this — if you see
30 m as the minimum, verify you are looking at the pipeline's output
and not an externally produced raster.

---

## Extending the pipeline

To add a new layer:

1. Add an entry under `layers:` in `config.yaml` with its path and weight.
2. Add a `_process_<name>` function to the appropriate module (or a new
   module if it is a new analysis type).
3. Register the module in `run_all.py`'s `MODULES` list.
4. Add the layer to `scan_layers` requirements if it needs its own gate.

To re-enable the statistics module:

Uncomment the `stats` entry in `MODULES` inside `run_all.py`.

---

## Environment

Pinned versions of the geospatial stack are managed via conda-forge
in `environment.yml`. The environment is Python 3.11; the key
dependencies are:

- `geopandas`, `shapely`, `rasterio`, `pyproj`, `fiona`
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `libpysal`, `esda` for the Getis-Ord Gi* statistic
- `pyyaml`, `tqdm`, `matplotlib`

To regenerate the environment:

```bash
conda env remove -n hec
conda env create -f environment.yml
```

---

## License and data handling

Internal use. Contact the project maintainer before redistribution.

Real input data must never be committed to this repository. The
`.gitignore` blocks `inputs/*` and every common geospatial extension;
if a source file must be shared, transfer it out-of-band.

---

## Citation

If this pipeline contributes to a publication, cite the project
repository and the version tag in use. Version is tracked in
`src/__init__.py` (`__version__`).
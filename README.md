# Hyperspectral Mars - Data Preprocessing & Training Guide

This repository contains scripts and notebooks for preparing Mars image data
and training models.

## Pipeline overview (from data to training)

1. CTX preprocessing (ISIS3) -> calibrated and map-projected CTX cubes
2. EVC creation (RGB proxy from CRISM bands to reduce training cost)
3. CRS reprojection and resampling
4. Coregistration
   - CTX–CRISM: AROSICS global + local
   - CaSSIS–CRISM: manual in QGIS (CRISM -> CaSSIS, saved as .tif)
5. CRISM to CaSSIS resampling (after manual coregistration)
6. Crop to CRISM valid bounds
7. Normalization (mean 0, std 1)
8. Training

## 1. CTX preprocessing (ISIS3)

The CTX preprocessing pipeline is implemented in:

- `preprocessing/ctx_preprocessing.sh`

### Prerequisites

- ISIS3 installed and available on PATH (required for CTX preprocessing).
  Installation guide: https://astrogeology.usgs.gov/docs/how-to-guides/environment-setup-and-maintenance/installing-isis-via-anaconda/
- A Bash shell to run the `.sh` scripts.
- An `equirectangular.map` file for `cam2map` (see CTX preprocessing step).
  These map files are installed automatically with ISIS and live under
  `/appdata/templates/maps`.

### What it does

For every `.img` file in the input folder, the script runs:

1. `mroctx2isis` to convert PDS `.img` to ISIS `.cub`
2. `spiceinit` to attach SPICE kernels
3. `ctxcal` to radiometrically calibrate the cube
4. `cam2map` to project to an equirectangular map projection

The outputs are written to the processed folder with these suffixes:

- `.cub`
- `.cal.cub`
- `.cal.eq.cub`

### Configure paths

Open `preprocessing/ctx_preprocessing.sh` and set:

- `CTX_FOLDER` - directory containing raw CTX `.img` files
- `CTX_PROCESSED_FOLDER` - directory to write processed `.cub` files
- `MAP_TEMP_FOLDER` - directory containing `equirectangular.map`

### Run

From the repository root on a system with ISIS3 installed:

```bash
bash preprocessing/ctx_preprocessing.sh
```

If you do not have an `equirectangular.map` file yet, create or copy one into
`MAP_TEMP_FOLDER` before running the script.

## 2. EVC creation

To reduce training cost, an Enhanced Visible Color (EVC) product is created
from three CRISM bands:

- 592 nm
- 533 nm
- 442 nm

Relevant script/notebook:

- `preprocessing/create_evc_util.py`
- `preprocessing/evc_creation.ipynb`

### Batch EVC creation (recommended path)

The most meaningful section of the notebook is **Batch EVC creation**. It
processes all hyperspectral cubes from `input_folder` and writes results to
`output_folder`. Each hyperspectral CRISM cube must be stored as two files:
Product ENVI Header `.hdr` and Product Data File `.img`.

## 3. CRS reprojection and resampling

Notebook: `crs_reprojection.ipynb`

- Warps the target dataset into the reference CRS.
- Resamples the target so target and reference share the same dimensions
  and pixel grid.


This is done with rasterio's virtual warping, which handles both reprojection
and resampling in one step. The reference image defines the output CRS,
resolution, and alignment; the target is reprojected onto that grid to keep
cropping and coregistration consistent. In this case, CRISM image is always the *target* image, and CaSSIS or CTX image is the *reference* image.

### CRISM to CaSSIS resampling (after manual coregistration)

This is where CaSSIS-CRISM pipeline deviates from CTX-CRISM pipeline, no reprojection and resampling for CaSSIS-CRISM need to be performed at this point, see next section.


## 4. Coregistration

Coregistration differs for CTX–CRISM vs CaSSIS–CRISM pairs.

### CTX–CRISM (AROSICS, automated)

Use `coregistration.ipynb` with AROSICS:
`https://git.gfz-potsdam.de/danschef/arosics`

Run both batch **global** coregistration (coarse, whole‑scene alignment) and
batch **local** coregistration (tile/window refinements).

Important: optimal parameters vary per pair. In AROSICS:

- `ws` = matching window size (larger is more robust, smaller is more local).
- `max_shift` = max allowed shift in pixels (limits search radius).

Adjust directories' paths accordingly as in previous steps.

### CaSSIS–CRISM (manual, QGIS)

Manual coregistration is performed in QGIS **first**, before any resampling or
cropping. The CaSSIS image remains the reference (unchanged); the CRISM image
is georeferenced to it and saved as a `.tif`. After this, run resampling, then
crop to CRISM valid bounds.

Steps in QGIS:

1. Add the reference layer (CaSSIS).
2. Open **Georeferencer**.
3. Import the target (CRISM) raster.
4. Pick ~10 control points.
5. Use **Thin Plate Spline** interpolation.
6. Use **Nearest Neighbor** resampling.

#### CRISM to CaSSIS resampling (after manual coregistration)

After manual coregistration, use the
**Reproject CRISM to CaSSIS** section and its **CRISM to CaSSIS resample batch**
subsection in `crs_reprojection.ipynb`.
Configure these paths in that cell:

- `evc_input_dir` (CRISM GeoTIFFs produced by manual coregistration)
- `cassis_input_dir` (CaSSIS ISIS cubes)
- `evc_resampled_dir` (output folder)

Before running the batch, ensure filenames are uppercase. The cell checks the
input folders and calls `files_in_dir_to_upper_case` from `utils.py` if any
lowercase letters are found (see the "Rename files to upper case" section in
the notebook for reference).

Note: this step marks the first use of CaSSIS data in the pipeline. In contrast to the other datasets, no specific preprocessing steps were necessary.

To carry out the same process for CTX CRISM pairs - reference the **CRISM to CTX reprojection** section

## 5. Crop to CRISM valid bounds

After reprojection/resampling, both datasets share the same grid but include
large non-overlapping areas. Crop both datasets to the CRISM valid bounds
(largest circumscribed rectangle of valid CRISM pixels) and apply the same
window to the aligned CaSSIS/CTX data.

Reference notebook: `crop_to_crism_bounds.ipynb`

Use the **Batch processing** section in that notebook. It takes:

- `evc_dir` (CRISM-EVC input folder)
- `evc_output_dir` (cropped output for CRISM-EVC)
- `cassis_dir` (CaSSIS or CTX input folder)
- `cassis_output_dir` (cropped output for CaSSIS/CTX)
- `pairs_dict` (mapping between CRISM and CaSSIS/CTX IDs)
- `bounds_fn` (function to compute CRISM valid bounds)

## 6. Normalization

`normalization.ipynb` normalizes EVC GeoTIFFs and ISIS cubes using percentile
clipping followed by per-band z-score normalization (mean 0, std 1).

Adjust these paths before running:

- `evc_dir` and `output_dir` in **Percentile clip and normalization for EVCs batch**
- `ctx_dir` and `output_dir` in **ISIS cubes files normalization batch**

Optional visualization sections let you inspect value distributions and
render a quick RGB preview.


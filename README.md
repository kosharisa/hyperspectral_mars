# Hyperspectral Mars - Data Preprocessing & Training Guide

This repository contains scripts and notebooks for preparing Mars image data
and training models.

## Pipeline overview (from data to training)

1. CTX preprocessing (ISIS3) -> calibrated and map-projected CTX cubes
2. CRISM preprocessing
   - CRS reprojection to match CTX
   - Coregistration (global/local or manual in QGIS)
3. EVC creation (RGB proxy from CRISM bands to reduce training cost)
4. Normalization (mean 0, std 1)
5. Training

## 1) CTX preprocessing (ISIS3)

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

## 2) EVC creation

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


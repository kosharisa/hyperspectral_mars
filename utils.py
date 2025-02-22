import rasterio
import numpy as np
import os

jezero_crater_pairs = {
    "FRT000047A3_07_IFEVCJ_MTR3": "P04_002743_1987_XI_18N282W",
    "FRT00005850_07_IFEVCJ_MTR3": "F04_037396_1985_XN_18N282W",
    "FRT00005C5E_07_IFEVCJ_MTR3": "F04_037396_1985_XN_18N282W",
    "FRT000066A4_07_IFEVCJ_MTR3": "P04_002743_1987_XI_18N282W",
    "FRT0001182A_07_IFEVCJ_MTR3": "B18_016786_1978_XN_17N283W",
    "FRT0001642E_07_IFEVCJ_MTR3": "B18_016509_1978_XN_17N282W",
    "FRT00017103_07_IFEVCJ_MTR3": "B19_016931_1975_XN_17N283W",
    "FRT0001ECBA_07_IFEVCJ_MTR3": "G13_023102_1986_XN_18N282W",
    "FRT0001FB74_07_IFEVCJ_MTR3": "G14_023669_1985_XN_18N282W",
    "FRT00021DA6_07_IFEVCJ_MTR3": "G14_023669_1985_XN_18N282W"
}

holden_crater_pairs = {
    "FRT0000474A_07_IF164J_MTR3" : "P04_002721_1538_XI_26S034W",
    "FRT00004F2F_07_IF164J_MTR3" : "P04_002721_1538_XI_26S034W",
    "FRT00006246_07_IF164J_MTR3" : "P04_002721_1538_XI_26S034W",
    "FRT00009172_07_IF164J_MTR3" : "P14_006690_1512_XI_28S034W",
    "FRT00009D17_07_IF164J_MTR3" : "P16_007191_1536_XI_26S035W",
    "FRT0000A98D_07_IF164J_MTR3" : "P18_008193_1536_XN_26S034W",
    "FRT0000ABB5_07_IF164J_MTR3" : "P19_008338_1531_XN_26S035W",
    "FRT0000B678_07_IF164J_MTR3" : "P22_009696_1531_XI_26S034W",
    "FRT0000BB9F_07_IF164J_MTR3" : "B02_010540_1532_XI_26S034W",
    "FRT0000C1D1_07_IF164J_MTR3" : "P22_009696_1531_XI_26S034W"
}




def calculate_valid_bounds(crism_path):
    """
    Calculates the valid bounding box of the CRISM image.
    """
    with rasterio.open(crism_path) as crism_ds:
        # Read the data as a mask, assuming the nodata value is set
        nodata = crism_ds.nodata
        crism_data = crism_ds.read(1)  # Read the first band (assuming it's representative)
        
        if nodata is None:
            raise ValueError("No nodata value found in CRISM image.")
        
        # Create a valid data mask
        valid_mask = crism_data != nodata

        # Get indices of valid data
        rows, cols = np.where(valid_mask)
        
        # Calculate bounding box in image coordinates
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        
        # Convert to geospatial bounds
        transform = crism_ds.transform
        valid_bounds = rasterio.transform.array_bounds(
            row_max - row_min + 1,
            col_max - col_min + 1,
            transform * rasterio.Affine.translation(col_min, row_min)
        )
        
        return valid_bounds
    

def percentile_clip(data, lower_percentile=0.01, upper_percentile=99.99):
    """
    Clips multichannel data using percentile clipping.
    
    Parameters:
        data (numpy.ndarray): The CRISM data array (height, width, bands).
        lower_percentile (float): The lower percentile (default is 0.01%).
        upper_percentile (float): The upper percentile (default is 99.99%).
    
    Returns:
        numpy.ndarray: The clipped CRISM data.
    """
    clipped_data = np.empty_like(data)
    
    lower_clip = np.nanpercentile(data, lower_percentile, axis=(1, 2), keepdims=True)
    upper_clip = np.nanpercentile(data, upper_percentile, axis=(1, 2), keepdims=True)

    clipped_data = np.clip(data, a_min=lower_clip, a_max=upper_clip)
    
    return clipped_data


def normalize(crism_data):

    normalized_cube = np.empty_like(crism_data, dtype=np.float32)

    for band in range(crism_data.shape[0]):
            mean = np.nanmean(crism_data[band])
            std = np.nanstd(crism_data[band])
            normalized_cube[band] = (crism_data[band] - mean) / std

    return normalized_cube


def create_mask(crism_data, no_data_value=65535):
    """
    Creates a binary mask for valid CRISM data, excluding no-value data.
    
    Args:
        crism_data (np.ndarray): The CRISM data array (3D or 2D).
        no_data_value (float or int, optional): Value indicating no data in the CRISM image.
                                               If None, the method will ignore this step.
    
    Returns:
        np.ndarray: A binary mask with 1 for valid pixels and 0 for invalid pixels.
    """
    
    mask = np.any(crism_data != no_data_value, axis=0)

    return mask.astype(bool)


def create_masked_data(data, no_data_value=65535):

    mask = create_mask(data, no_data_value)
    return np.where(mask, data, np.nan)


def get_files_from_dir(dir, file_ext):
    """
    Returns a list of files with the specified extension in the directory.

    Args:
        dir (str): The directory path.
        file_ext (str): The file extension to filter by.
    """
    return sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(file_ext)])


def files_in_dir_to_upper_case(path):
    """
    Changes the case of all files in the specified directory to upper case. keeping the file extension in lower.
    """
    for file in os.listdir(path):
        os.rename(os.path.join(path, file), os.path.join(path, f"{file.split('.')[0].upper()}.{file.split('.')[-1]}"))


def copy_crs_from_ctx(crism_path, ctx_path):
    """
    ! USE ONLY AFTER REPROJECTION, NOT A SUBSTITUTION FOR REPROJECTION !

    Updates the coordinate system string in an ENVI header file to match the CTX projection.

    Args:
        crism_path (str): Path to the CRISM image to update the projection.
        ctx_path (str): Path to the CTX image to extract correct projection.
    """
    header_path = find_crism_header(crism_path)
    # Extract correct projection from CTX
    with rasterio.open(ctx_path) as ctx_src:
        correct_wkt = ctx_src.crs.to_wkt()  # Convert CRS to WKT format
    
    # Read the ENVI header file
    with open(header_path, "r") as file:
        header_content = file.read()

    # Regex pattern to replace the coordinate system string
    pattern = r'(coordinate system string\s*=\s*{)(.*?)(})'
    
    # Replace with correct WKT from CTX
    corrected_header = re.sub(pattern, rf'\1{correct_wkt}\3', header_content, flags=re.DOTALL)

    # Write back to the header file
    with open(header_path, "w") as file:
        file.write(corrected_header)

    print(f"Updated coordinate system in: {header_path}")


def get_base_name(file_name):
    """
    Returns the base name of a file without the extension.
    """
    return file_name.split('.')[0]


def split_path_into_dir_and_file(path):
    """
    Splits a path into the directory and file name.
    """
    directory, file = os.path.split(path)
    return directory, file


def find_crism_header(crism_path):
    """
    Finds the ENVI header file for a CRISM image.
    """
    crism_dir, crism_file = split_path_into_dir_and_file(crism_path)
    header_file = f"{os.path.splitext(crism_file)[0]}.hdr"
    header_path = os.path.join(crism_dir, header_file)
    return header_path


def normalize_band(band):
    band_min, band_max = np.nanmin(band), np.nanmax(band)
    band_norm = (band - band_min) / (band_max - band_min) * 255
    return band_norm.astype(np.uint8)


def apply_3_sigma_stretch(band):
    mean = np.nanmean(band)
    std = np.nanstd(band)
    min_val = mean - 3 * std
    max_val = mean + 3 * std
    band = np.clip(band, min_val, max_val)
    return band


def create_np_image(data, sigma_stretch=False):
    bands = []
    for band in data:
        if sigma_stretch:
            band = apply_3_sigma_stretch(band)
        band = normalize_band(band)
        bands.append(band)
    return np.stack(bands, axis=0)
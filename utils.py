import rasterio
import numpy as np
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

jezero_crater_pairs = {
    "FRT000047A3_07_IFEVCJ_MTR3": "P04_002743_1987_XI_18N282W",
    "FRT00005850_07_IFEVCJ_MTR3": "F04_037396_1985_XN_18N282W",
    "FRT00005C5E_07_IFEVCJ_MTR3": "F04_037396_1985_XN_18N282W",
    "FRT000066A4_07_IFEVCJ_MTR3": "P04_002743_1987_XI_18N282W",
    "FRT0001182A_07_IFEVCJ_MTR3": "B18_016786_1978_XN_17N283W",
    "FRT0001642E_07_IFEVCJ_MTR3": "B18_016509_1978_XN_17N282W", # val
    "FRT00017103_07_IFEVCJ_MTR3": "B19_016931_1975_XN_17N283W",
    "FRT0001ECBA_07_IFEVCJ_MTR3": "G13_023102_1986_XN_18N282W", # test
    "FRT0001FB74_07_IFEVCJ_MTR3": "G14_023669_1985_XN_18N282W",
    "FRT00021DA6_07_IFEVCJ_MTR3": "G14_023669_1985_XN_18N282W"
}

jezero_crism_cassis_pairs = {
    "FRT00005850_07_IFEVCJ_MTR3": "MY36_017709_165_0", 
    "FRT00005C5E_07_IFEVCJ_MTR3": "MY36_020142_020_3",
    "FRT000047A3_07_IFEVCJ_MTR3": "MY36_020142_020_3", 
    "FRT000066A4_07_IFEVCJ_MTR3": "MY36_016465_161_0", 
    "FRT0001182A_07_IFEVCJ_MTR3": "no data",
    "FRT0001642E_07_IFEVCJ_MTR3": "MY37_024061_018_0", # val
    "FRT00017103_07_IFEVCJ_MTR3": "MY36_020471_161_1", 
    "FRT0001ECBA_07_IFEVCJ_MTR3": "MY35_009664_162_0", # test
    "FRT0001FB74_07_IFEVCJ_MTR3": "no_data",
    "FRT00021DA6_07_IFEVCJ_MTR3": "MY36_016465_161_0" 
}

holden_crater_pairs = {
    "FRT0000474A_07_IFEVCJ_MTR3" : "P04_002721_1538_XI_26S034W", # test
    "FRT00004F2F_07_IFEVCJ_MTR3" : "P04_002721_1538_XI_26S034W",
    "FRT00006246_07_IFEVCJ_MTR3" : "P04_002721_1538_XI_26S034W",
    "FRT00009172_07_IFEVCJ_MTR3" : "P14_006690_1512_XI_28S034W",
    "FRT00009D17_07_IFEVCJ_MTR3" : "P16_007191_1536_XI_26S035W",
    "FRT0000A98D_07_IFEVCJ_MTR3" : "P18_008193_1536_XN_26S034W",
    "FRT0000ABB5_07_IFEVCJ_MTR3" : "P19_008338_1531_XN_26S035W", # val
    "FRT0000B678_07_IFEVCJ_MTR3" : "P22_009696_1531_XI_26S034W",
    "FRT0000BB9F_07_IFEVCJ_MTR3" : "B02_010540_1532_XI_26S034W",
    "FRT0000C1D1_07_IFEVCJ_MTR3" : "P22_009696_1531_XI_26S034W"
}




def calculate_valid_bounds(crism_path):
    """
    Calculates the valid bounding box of the CRISM image.
    """
    with rasterio.open(crism_path) as crism_ds:
        # Read the data as a mask, assuming the nodata value is set
        nodata = crism_ds.nodata
        crism_data = crism_ds.read()  # Read the first band (assuming it's representative)
        
        if nodata is None:
            raise ValueError("No nodata value found in CRISM image.")
        
        # Create a valid data mask
        valid_mask = np.all(crism_data != nodata, axis=0)

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
    

def calculate_only_valid_bounds(crism_path):
    """
    Finds the largest inscribed rectangle within valid CRISM pixels.

    Args:
        crism_path (str): Path to the CRISM image.

    Returns:
        tuple: Bounding box coordinates in the format (min_x, min_y, max_x, max_y)
    """
    with rasterio.open(crism_path) as crism_ds:
        crism_data = crism_ds.read()  # Read first band (assumes all bands share the same mask)
        transform = crism_ds.transform  # Get affine transformation

        # Create a binary mask (1 = valid, 0 = invalid)
        mask = create_mask(crism_data)

        rows = len(mask)
        cols = len(mask[0])
        max_area = 0
        max_coords = (0, 0, 0, 0)
        heights = [0] * cols  # Tracks the height of consecutive valid pixels for each column
    
        for row_idx in range(rows):
            # Update heights for each column based on current row
            for col in range(cols):
                heights[col] = heights[col] + 1 if mask[row_idx][col] else 0
        
            stack = []
            current_max_area = 0
            current_coords = (0, 0, 0, 0)
        
            # Process each column to calculate the largest rectangle in the histogram
            for i in range(cols + 1):
                current_height = heights[i] if i < cols else 0  # Use 0 to trigger final pops
            
                while stack and current_height < heights[stack[-1]]:
                    popped_idx = stack.pop()
                    height = heights[popped_idx]
                    left_bound = stack[-1] if stack else -1
                    width = i - left_bound - 1
                
                    if width <= 0:
                        continue  # Skip invalid width
                
                    area = height * width
                    if area > current_max_area:
                        current_max_area = area
                        left_col = left_bound + 1
                        right_col = i - 1
                        # Ensure the rectangle is within valid rows
                        top_row = row_idx - height + 1
                        bottom_row = row_idx
                        current_coords = (left_col, top_row, right_col, bottom_row)
            
                # Push the current column index to the stack
                if i < cols:
                    stack.append(i)
        
            # Update global maximum if current row's maximum is larger
            if current_max_area > max_area:
                max_area = current_max_area
                max_coords = current_coords
    
        # Ensure coordinates are within bounds (for edge cases)
        col_min, row_min, col_max, row_max = max_coords
        col_min = max(0, min(col_min, cols - 1))
        col_max = max(0, min(col_max, cols - 1))
        row_min = max(0, min(row_min, rows - 1))
        row_max = max(0, min(row_max, rows - 1))

        # Extract row/column bounds
        # row_min, col_min, row_max, col_max = max_coords

        # Convert to geospatial coordinates
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
    Creates a binary mask for valid CRISM data, excluding no-value data and NaNs.
    
    Args:
        crism_data (np.ndarray): The CRISM data array (3D or 2D).
        no_data_value (float or int, optional): Value indicating no data in the CRISM image.
                                               If None, the method will ignore this step.
    
    Returns:
        np.ndarray: A binary mask with 1 for valid pixels and 0 for invalid pixels.
    """
    # Mask for no_data_value
    mask = np.all(crism_data != no_data_value, axis=0)
    nan_mask = ~np.isnan(crism_data).all(axis=0)
    # Combine masks
    combined_mask = mask & nan_mask

    return combined_mask.astype(bool)


def create_masked_data(data, no_data_value=65535):

    mask = create_mask(data, no_data_value)
    return np.where(mask, data, np.nan)


def create_cassis_crism_combined_mask(cassis_data, crism_data, cassis_no_data_value=None, crism_no_data_value=None):
    """
    Creates a combined mask for Cassis and CRISM data, ensuring both datasets have valid pixels.
    
    Args:
        cassis_data (np.ndarray): The Cassis data array (3D or 2D).
        crism_data (np.ndarray): The CRISM data array (3D or 2D).
    
    Returns:
        np.ndarray: A binary mask with 1 for pixels valid in both datasets and 0 otherwise.
    """
    cassis_mask = create_mask(cassis_data, cassis_no_data_value)
    crism_mask = create_mask(crism_data, crism_no_data_value)
    
    combined_mask = cassis_mask & crism_mask
    return combined_mask.astype(bool)


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
    """
    Applies a 3-sigma stretch to the data.
    """
    mean = np.nanmean(band)
    std = np.nanstd(band)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    return np.clip(band, lower_bound, upper_bound)


def create_np_image(data, mask=None, sigma_stretch=False):
    bands = []
    if mask is not None:
        data = np.where(mask, data, np.nan)
    for band in data:
        if sigma_stretch:
            band = apply_3_sigma_stretch(band)
        band = normalize_band(band)
        bands.append(band)
    return np.stack(bands, axis=0)


def calculate_rmse(prediction, target, channel_wise=False):
    """
    Calculates the root mean squared error between the prediction and target.
    Images' shape (H, W, C)
    """

    mask = create_mask(target.transpose(2, 0, 1))
    mask = np.broadcast_to(mask, target.shape[:2])  # Expand 2D mask to (H, W, C)
    mask = np.broadcast_to(mask[..., np.newaxis], target.shape)

    # Mask valid pixels (flatten for easy indexing)
    valid_target = target[mask].reshape(-1)
    valid_prediction = prediction[mask].reshape(-1)
    
    # Calculate squared error on VALID PIXELS
    squared_error = (valid_target - valid_prediction) ** 2
    
    # Reshape for channel-wise computation
    if channel_wise and target.ndim == 3:
        num_channels = target.shape[-1]
        squared_error = squared_error.reshape(-1, num_channels)
        mse = np.nanmean(squared_error, axis=0)
    else:
        mse = np.nanmean(squared_error)
    
    rmse = np.sqrt(mse)

    min_val, max_val = np.min(valid_target), np.max(valid_target)
    pixel_range = max_val - min_val if (max_val - min_val) > 0 else 1e-8

    # Scale RMSE to [0, 1]
    # scaled_rmse = rmse / pixel_range

    percent_rmse = (1.0 - (rmse / pixel_range)) * 100
    return rmse, percent_rmse


def calculate_cosine_similarity(prediction, target, mask=None):
        # If mask is not provided, create mask from target
    if mask is None:
        mask = create_mask(target.transpose(2, 0, 1))
        mask = np.broadcast_to(mask, target.shape[:2])
        mask = np.broadcast_to(mask[..., np.newaxis], target.shape)

    # Mask valid pixels
    valid_pred = prediction[mask].reshape(-1, target.shape[-1])
    valid_target = target[mask].reshape(-1, target.shape[-1])

    # If only one channel, reshape to 2D
    if valid_pred.ndim == 1:
        valid_pred = valid_pred.reshape(-1, 1)
        valid_target = valid_target.reshape(-1, 1)

    # Compute cosine similarity for each channel
    similarities = []
    for i in range(valid_pred.shape[1]):
        pred_channel = valid_pred[:, i]
        target_channel = valid_target[:, i]
        valid_idx = np.isfinite(pred_channel) & np.isfinite(target_channel)
        if np.sum(valid_idx) == 0:
            similarities.append(np.nan)
            continue
        pred_vec = pred_channel[valid_idx].reshape(1, -1)
        target_vec = target_channel[valid_idx].reshape(1, -1)
        sim = cosine_similarity(pred_vec, target_vec)[0, 0]
        similarities.append(sim)


    mean_similarity = np.nanmean(similarities)
    return similarities, mean_similarity
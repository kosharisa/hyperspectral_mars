import numpy as np
import rasterio
import spectral as sp


def create_evc(source_path, output_path, evc_bands):
    img = sp.open_image(source_path)
    rgb_slices = img[:, :, evc_bands]

    hdr = __adjust_header(img.metadata, evc_bands)

    sp.envi.save_image(output_path, rgb_slices, metadata=hdr)

    __clean_header(output_path, output_path)


def __adjust_header(hdr, bands):

    hdr['bands'] = len(bands)

    for key in ['wavelength', 'fwhm', 'bbl']:
        if key in hdr:
            values = hdr[key]
            hdr[key] = [values[b] for b in bands]

    return hdr


# is necessary because of the way sp writes envi header, which is not suitable for further preprocessing
def __clean_header(source_path, output_path):
    with open(source_path, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        # Remove spaces before commas and around curly braces
        line = line.replace(' ,', ',').replace('{ ', '{').replace(' }', '}')
        cleaned_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(cleaned_lines)
    

def save_npy_as_img(data, output_path):
    height, width, bands = data.shape
    dtype = data.dtype

    data = np.moveaxis(data, -1, 0)

    metadata = {
        "driver": "ENVI",          # File format
        "height": height,          # Image height (rows)
        "width": width,            # Image width (columns)
        "count": bands,            # Number of bands
        "dtype": dtype.name,       # Data type (e.g., float32, int16)
    }
    
    with rasterio.open(output_path, 'w', **metadata) as dst:
        for i in range(data.shape[0]):
            dst.write(data[i, :, :], indexes=i + 1)

import numpy as np
from astropy.io import fits

obs_file = "test_xfw/v_offset_inner.fits"

try:
    with fits.open(obs_file) as hdul:
        data = hdul[0].data
        if data.ndim > 2:
            data = data.squeeze()
        print(f"Observation Shape: {data.shape}")
        
        # Check if we can get pixel scale to suggest npix
        header = hdul[0].header
        # Assuming CDELT1/2 are in degrees
        cdelt1 = abs(header.get('CDELT1', 0))
        cdelt2 = abs(header.get('CDELT2', 0))
        
        print(f"CDELT1: {cdelt1}, CDELT2: {cdelt2}")
        
except Exception as e:
    print(f"Error checking obs: {e}")

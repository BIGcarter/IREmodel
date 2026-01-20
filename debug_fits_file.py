import os
from astropy.io import fits

mask_file = "test_xfw/mask.fits"

print(f"Checking file: {mask_file}")
if not os.path.exists(mask_file):
    print("Error: File does not exist.")
else:
    file_size = os.path.getsize(mask_file)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print("Error: File is empty (0 bytes).")
    else:
        try:
            print("Attempting to read with fits.open()...")
            with fits.open(mask_file, ignore_missing_simple=True) as hdul:
                hdul.info()
                print("Read successful.")
        except Exception as e:
            print(f"Failed to read FITS: {e}")
            
        # Hex dump first few bytes to check header
        print("\nFirst 80 bytes (Hex):")
        with open(mask_file, 'rb') as f:
            header_bytes = f.read(80)
            print(header_bytes)

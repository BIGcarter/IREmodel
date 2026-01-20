import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings

def plot_moment(filename, moment_type='mom0', output_filename=None):
    """
    Plot Moment-0 or Moment-1 map.
    
    Args:
        filename (str): FITS file path.
        moment_type (str): 'mom0' or 'mom1'.
        output_filename (str): Output image file path (e.g. 'plot.png').
    """
    if not filename.endswith('.fits'):
        filename += '.fits'
        
    try:
        with fits.open(filename) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            # Flip X-axis back if needed for display?
            # FITS data is (ny, nx) usually.
            # Usually plt.imshow displays with origin='lower'.
            # If we want RA increasing to left, we need to check extent or flip.
            
            # Simple plot
            plt.figure(figsize=(10, 8))
            
            if moment_type == 'mom0':
                cmap = 'inferno'
                label = 'Integrated Intensity (Jy/beam km/s)'
                vmin = np.nanmin(data)
                vmax = np.nanmax(data)
            else:
                cmap = 'RdBu_r'
                label = 'Velocity (km/s)'
                # Center colormap around system velocity?
                # Usually mean is close to 0 if subtracted vsys.
                abs_max = np.nanmax(np.abs(data))
                vmin = -abs_max
                vmax = abs_max
            
            plt.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label=label)
            
            title = header.get('OBJECT', 'Unknown Object')
            plt.title(f"{title} - {moment_type}")
            plt.xlabel("RA Offset (pixel)")
            plt.ylabel("Dec Offset (pixel)")
            
            if output_filename:
                plt.savefig(output_filename)
                print(f"Saved plot: {output_filename}")
            else:
                plt.show()
            plt.close()
            
    except Exception as e:
        print(f"Error plotting {filename}: {e}")

def plot_pv(filename, output_filename=None):
    """
    Plot PV diagram.
    """
    if not filename.endswith('.fits'):
        filename += '.fits'
        
    try:
        with fits.open(filename) as hdul:
            data = hdul[0].data # (nvel, npix_pv)
            header = hdul[0].header
            
            # Data usually (Velocity, Offset)
            
            plt.figure(figsize=(10, 6))
            
            # Aspect ratio might be weird, so set auto
            plt.imshow(data, origin='lower', cmap='inferno', aspect='auto')
            plt.colorbar(label='Intensity (Jy/beam)')
            
            title = header.get('OBJECT', 'Unknown Object')
            plt.title(f"{title} - PV Diagram")
            plt.xlabel("Position Offset (pixel)")
            plt.ylabel("Velocity (pixel)")
            
            if output_filename:
                plt.savefig(output_filename)
                print(f"Saved plot: {output_filename}")
            else:
                plt.show()
            plt.close()
            
    except Exception as e:
        print(f"Error plotting {filename}: {e}")

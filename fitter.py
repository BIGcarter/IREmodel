import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import itertools
from models import IREModel
from params import SourceParams
from config import CMPERKM

class ModelFitter:
    def __init__(self, obs_mom1_path, rms_noise=None, threshold=0.01, mask_path=None, region_path=None):
        """
        Initialize the fitter with observational data.
        
        Args:
            obs_mom1_path (str): Path to the observational Moment-1 FITS file.
            rms_noise (float, optional): RMS noise level for Chi-square calculation.
                                         If None, calculates MSE instead.
            threshold (float): Relative threshold (0.0-1.0) to mask weak signals in observation
                               if no explicit mask is provided in the FITS.
            mask_path (str, optional): Path to a FITS file containing a mask (1=valid, 0=invalid).
            region_path (str, optional): Path to a DS9 region file (.reg) to create a mask.
        """
        self.obs_mom1_path = obs_mom1_path
        self.rms_noise = rms_noise
        
        # Read Observation
        with fits.open(obs_mom1_path) as hdul:
            # Assume data is in extension 0
            self.obs_data = hdul[0].data
            self.obs_header = hdul[0].header
            
            # Squeeze if needed (e.g. if shape is (1, ny, nx))
            if self.obs_data.ndim > 2:
                self.obs_data = self.obs_data.squeeze()
                
            # Handle NaNs (common in Mom1 maps)
            self.obs_mask = ~np.isnan(self.obs_data)
            
            # Load explicit mask from FITS if provided
            if mask_path:
                with fits.open(mask_path, ignore_missing_simple=True) as m_hdul:
                    mask_data = m_hdul[0].data
                    if mask_data.ndim > 2:
                        mask_data = mask_data.squeeze()
                    
                    if mask_data.shape != self.obs_data.shape:
                        print(f"Warning: Mask shape {mask_data.shape} mismatch with Obs {self.obs_data.shape}. Resizing not implemented.")
                    else:
                        # Combine with NaN mask
                        explicit_mask = (mask_data > 0.5)
                        self.obs_mask = self.obs_mask & explicit_mask
                        print(f"Loaded mask from {mask_path}. Valid pixels: {np.sum(self.obs_mask)}")
            
            # Load mask from Region file if provided (DS9 or CRTF)
            if region_path:
                try:
                    import regions
                    from astropy.wcs import WCS
                    
                    # Read region file
                    # Auto-detect format or try DS9 then CRTF
                    # The file might be named .fits but contain text (CRTF/DS9)
                    
                    try:
                        regs = regions.Regions.read(region_path, format='ds9')
                    except:
                        try:
                            regs = regions.Regions.read(region_path, format='crtf')
                        except:
                            # Fallback: try auto detection if possible or raise
                            print("Could not determine region format (DS9/CRTF).")
                            raise
                    
                    # Create WCS object from header
                    wcs = WCS(self.obs_header)
                    # If obs_data was squeezed (e.g. from 3D to 2D), WCS might need slicing
                    # Assuming standard Mom1 FITS with 2D WCS or compatible 3D WCS
                    if wcs.naxis > 2:
                        wcs = wcs.dropaxis(2) # Drop freq/vel axis if present
                    
                    # Create empty mask
                    ny, nx = self.obs_data.shape
                    combined_region_mask = np.zeros((ny, nx), dtype=bool)
                    
                    print(f"Loading regions from {region_path}...")
                    for i, reg in enumerate(regs):
                        # Convert to pixel region
                        pix_reg = reg.to_pixel(wcs)
                        # Create mask for this region
                        # mode='center' is faster, 'exact' is more precise. 'center' is usually fine for fitting.
                        mask_i = pix_reg.to_mask(mode='center')
                        
                        # Add to full mask
                        # to_image(shape) puts the mask back into the full array
                        combined_region_mask = combined_region_mask | mask_i.to_image((ny, nx)).astype(bool)
                        
                    # Apply to obs_mask
                    # Region defines VALID area (True inside)
                    self.obs_mask = self.obs_mask & combined_region_mask
                    print(f"Applied region mask. Valid pixels: {np.sum(self.obs_mask)}")
                    
                except ImportError:
                    print("Error: 'regions' package not installed. Cannot parse .reg file.")
                except Exception as e:
                    print(f"Error parsing region file: {e}")
            
            # If rms is provided, we can compute Chi2
            # Chi2 = Sum( (Obs - Model)^2 / RMS^2 )
            
            # Read geometry from header to ensure model matches?
            # Ideally, we pass these to SourceParams.
            # But for now, we assume the user provides a template params_dict
            # that matches the observation's geometry (pixel scale, beam, etc).
            
    def loss_function(self, model_mom1):
        """
        Calculate loss (MSE or Chi2) between Model and Observation.
        """
        # Ensure shapes match
        # model_mom1 from SkyPlane.calculate_moments is (nx, ny)
        # FERIA C++ and Python io_utils.py flip the X-axis (RA) when writing to FITS
        # To match Observation (which is a FITS file), we must apply the same transformation
        # 1. Transpose to (ny, nx)
        model_transformed = model_mom1.T
        # 2. Flip X-axis (RA) to match FITS convention used in FERIA
        model_transformed = model_transformed[:, ::-1]
            
        if model_transformed.shape != self.obs_data.shape:
             # Handle shape mismatch by center cropping/padding
             # Assuming both are centered on the source
             obs_ny, obs_nx = self.obs_data.shape
             mod_ny, mod_nx = model_transformed.shape
             
             if mod_ny >= obs_ny and mod_nx >= obs_nx:
                 # Crop Model to match Obs
                 start_y = (mod_ny - obs_ny) // 2
                 start_x = (mod_nx - obs_nx) // 2
                 model_transformed = model_transformed[start_y:start_y+obs_ny, start_x:start_x+obs_nx]
             else:
                 # Model is smaller? This shouldn't happen if we set npix large enough.
                 # If it happens, raise error or pad model?
                 raise ValueError(f"Shape mismatch: Obs {self.obs_data.shape} vs Model {model_transformed.shape}. Model must be larger or equal.")

        # Calculate Residuals on valid pixels (where Obs is not NaN)
        # Also maybe mask Model where it is zero?
        # Note: True Obs was generated with plot=True, which saves to disk.
        # But here we read it back.
        
        # Check alignment using a debug print if loss is high
        # diff_sum = np.sum(np.abs(self.obs_data - model_transformed))
        # print(f"Diff Sum: {diff_sum}")
        
        valid_mask = self.obs_mask & (np.abs(model_transformed) > 1e-6)
        
        diff = self.obs_data[valid_mask] - model_transformed[valid_mask]
        
        if self.rms_noise:
            chi2 = np.sum(np.square(diff / self.rms_noise))
            return chi2 / np.sum(valid_mask) # Reduced Chi2 approximation (per pixel)
        else:
            mse = np.mean(np.square(diff))
            return mse

def fit_grid(obs_file, template_params, param_grid, output_dir="fit_output", npix=128, nvel=64, mask_file=None, region_file=None):
    """
    Perform grid search fitting.
    
    Args:
        obs_file (str): Path to observation FITS (Mom1).
        template_params (dict): Base parameters for the model.
        param_grid (dict): Dictionary of varying parameters (e.g. {'mass': [1,2], 'inc': [30,40]}).
        output_dir (str): Directory to save plots.
        mask_file (str, optional): Path to a FITS mask file.
        region_file (str, optional): Path to a DS9 region file (.reg).
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fitter = ModelFitter(obs_file, mask_path=mask_file, region_path=region_file)
    
    # Prepare Grid
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Starting Grid Search on {len(combinations)} models...")
    print(f"Varying Parameters: {keys}")
    
    results = []
    
    for i, combo in enumerate(combinations):
        # Update params
        current_params_dict = template_params.copy()
        for k, v in zip(keys, combo):
            current_params_dict[k] = v
            
        # Create Model
        try:
            params = SourceParams(**current_params_dict)
            model = IREModel(params, npix=npix, nvel=nvel)
            
            # Run Model (No File I/O)
            model.make_cube(output_filename=None) # Pure memory
            mom0, mom1 = model.make_moments(output_prefix=None, moment_type='mom1', plot=False)
            
            # Calculate Loss
            loss = fitter.loss_function(mom1)
            
            results.append(tuple(list(combo) + [loss]))
            
            print(f"[{i+1}/{len(combinations)}] Params: {combo} -> Loss: {loss:.4f}")
            
        except Exception as e:
            print(f"Error fitting model {combo}: {e}")
            results.append(tuple(list(combo) + [np.nan]))

    # Convert results to structured array or pandas df
    # For simplicity, just lists
    
    # Visualization
    if len(keys) == 1:
        # 1D Plot
        x = [r[0] for r in results]
        y = [r[-1] for r in results]
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'o-', linewidth=2)
        plt.xlabel(keys[0])
        plt.ylabel("Loss (MSE/Chi2)")
        plt.title(f"1D Fit: {keys[0]}")
        plt.grid(True)
        out_path = f"{output_dir}/fit_1d_{keys[0]}.png"
        plt.savefig(out_path)
        print(f"Saved 1D plot: {out_path}")
        
    elif len(keys) == 2:
        # 2D Heatmap / Corner-like
        # We need to reshape results into grid
        
        # Unique values for axes
        x_vals = np.unique([r[0] for r in results]) # Key 0
        y_vals = np.unique([r[1] for r in results]) # Key 1
        
        loss_grid = np.zeros((len(y_vals), len(x_vals)))
        
        # Fill grid
        # Assuming product order matches
        for r in results:
            xi = np.where(x_vals == r[0])[0][0]
            yi = np.where(y_vals == r[1])[0][0]
            loss_grid[yi, xi] = r[2]
            
        plt.figure(figsize=(10, 8))
        plt.imshow(loss_grid, origin='lower', aspect='auto', cmap='viridis_r',
                   extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()])
        plt.colorbar(label='Loss')
        plt.xlabel(keys[0])
        plt.ylabel(keys[1])
        plt.title(f"2D Fit: {keys[0]} vs {keys[1]}")
        
        # Mark minimum
        min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
        best_x = x_vals[min_idx[1]]
        best_y = y_vals[min_idx[0]]
        plt.plot(best_x, best_y, 'r*', markersize=15, label=f'Best: {best_x}, {best_y}')
        plt.legend()
        
        out_path = f"{output_dir}/fit_2d_{keys[0]}_{keys[1]}.png"
        plt.savefig(out_path)
        print(f"Saved 2D plot: {out_path}")
        
    else:
        # High-dimensional plotting using corner.py
        try:
            import corner
            
            # Prepare data for corner plot
            # Corner plots usually show probability distributions P ~ exp(-0.5 * chi2)
            # Or just samples weighted by likelihood.
            # But we have grid samples.
            
            # We can treat this as discrete samples.
            # Or just plot the loss as a color?
            # Standard corner plots are for MCMC chains.
            
            # Alternative: Plot weighted corner plot
            # Weights = exp(-Loss / (2*min_loss?)) or just 1/Loss
            
            data = np.array([r[:-1] for r in results]) # Parameters
            losses = np.array([r[-1] for r in results])
            
            # Simple weighting: exp(-0.5 * (Loss - min_loss)) if Loss is Chi2
            # If Loss is MSE, scaling is arbitrary.
            # Let's just use 1/Loss for visualization or exp(-Loss)
            
            # For grid search visualization, usually we just want to see where the minimum is.
            # Let's compute a "likelihood"
            min_loss = np.nanmin(losses)
            # Avoid overflow, assume Loss is somewhat like Chi2
            # If Loss is very large, likelihood -> 0
            
            # Heuristic scaling for visibility
            # Normalize loss to 0-1 range?
            # weights = np.exp(-(losses - min_loss))
            
            # Actually, corner.corner expects samples.
            # We can resample based on weights to generate a "chain"
            
            # Better approach for grid: just plot all points but color by loss?
            # corner.py doesn't support scatter color easily in histograms.
            
            # Let's generate a "best fit" corner plot by resampling
            # This is a bit hacky but gives the visual representation of probability
            
            # Assume Loss ~ Chi2
            # Likelihood L ~ exp(-0.5 * Loss)
            # If Loss is MSE, we need an estimate of sigma^2 to convert to Chi2
            # Chi2 = N_pix * MSE / sigma^2
            # Let's assume sigma approx sqrt(MSE_min) for scaling?
            
            # Let's use a simpler approach: 
            # Weighted samples
            weights = np.exp(-0.5 * (losses - min_loss))
            
            # Resample
            # Create a large number of samples
            n_samples = 10000
            probs = weights / np.sum(weights)
            indices = np.random.choice(len(losses), size=n_samples, p=probs)
            samples = data[indices]
            
            fig = corner.corner(samples, labels=keys, show_titles=True, 
                                title_fmt=".2f", plot_datapoints=False, fill_contours=True)
            
            fig.suptitle(f"Corner Plot (Resampled from Grid, Min Loss={min_loss:.4f})", fontsize=14)
            
            out_path = f"{output_dir}/fit_corner.png"
            fig.savefig(out_path)
            print(f"Saved Corner plot: {out_path}")
            
        except ImportError:
            print("Error: 'corner' package not installed. Skipping corner plot.")
        except Exception as e:
            print(f"Error plotting corner: {e}")

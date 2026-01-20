import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import itertools
from models import IREModel
from params import SourceParams
from config import CMPERKM

class ModelFitter:
    def __init__(self, obs_mom1_path, rms_noise=None, threshold=0.01):
        """
        Initialize the fitter with observational data.
        
        Args:
            obs_mom1_path (str): Path to the observational Moment-1 FITS file.
            rms_noise (float, optional): RMS noise level for Chi-square calculation.
                                         If None, calculates MSE instead.
            threshold (float): Relative threshold (0.0-1.0) to mask weak signals in observation
                               if no explicit mask is provided in the FITS.
                               Actually, we should use the model's intensity mask?
                               Or mask the observation based on its own intensity?
                               Assuming obs_mom1 is already masked or we use a simple region.
                               Here we just read the data.
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
        if model_mom1.shape != self.obs_data.shape:
            # model_mom1 from SkyPlane.calculate_moments is (nx, ny)
            # FITS data is (ny, nx) usually
            # FERIA C++ and Python io_utils.py flip the X-axis (RA) when writing to FITS
            # To match Observation (which is a FITS file), we must apply the same transformation
            # 1. Transpose to (ny, nx)
            model_transformed = model_mom1.T
            # 2. Flip X-axis (RA) to match FITS convention used in FERIA
            model_transformed = model_transformed[:, ::-1]
        else:
            # If shape matches, assume it's already transformed?
            # But here model_mom1 comes directly from memory (nx, ny).
            # If obs_data is (128, 128) and model is (128, 128), 
            # we STILL need to transpose/flip because internal axes != FITS axes.
            model_transformed = model_mom1.T
            model_transformed = model_transformed[:, ::-1]
            
        if model_transformed.shape != self.obs_data.shape:
             raise ValueError(f"Shape mismatch: Obs {self.obs_data.shape} vs Model {model_transformed.shape}")

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

def fit_grid(obs_file, template_params, param_grid, output_dir="fit_output", npix=128, nvel=64):
    """
    Perform grid search fitting.
    
    Args:
        obs_file (str): Path to observation FITS (Mom1).
        template_params (dict): Base parameters for the model.
        param_grid (dict): Dictionary of varying parameters (e.g. {'mass': [1,2], 'inc': [30,40]}).
        output_dir (str): Directory to save plots.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fitter = ModelFitter(obs_file)
    
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
        print("High-dimensional plotting not implemented yet (requires corner.py).")
        # Could save results to CSV

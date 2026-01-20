import numpy as np
import os
import itertools
from config import MSUN, GRAV, CMPERAU, CMPERKM, DEGPERRAD, NO_GAS, IS_IRE, IS_KEP, EPS
from params import SourceParams
from mesh import Mesh
from sky import SkyPlane
from pv_diagram import PVDiagram
from io_utils import write_fits_cube, write_fits_pv, write_fits_moment
from plot_utils import plot_moment, plot_pv

class IREModel:
    def __init__(self, params: SourceParams, npix=128, nvel=64):
        self.params = params
        self.npix = npix
        self.nvel = nvel
        
        # Convert units and store parameters
        self.mass_grav = params.mass_msun * MSUN * GRAV
        self.r_cb = params.rcb_au * CMPERAU
        self.inc = params.inc_deg / DEGPERRAD
        self.pa = params.pa_deg / DEGPERRAD
        self.rot = params.rot_sign
        self.r_out = params.rout_au * CMPERAU
        self.r_in = params.rin_au * CMPERAU
        
        self.height_ire = params.height_ire_au * CMPERAU
        self.tan_flare_ire = np.tan(params.flare_ire_deg / DEGPERRAD / 2.0)
        
        self.height_kep = params.height_kep_au * CMPERAU
        self.tan_flare_kep = np.tan(params.flare_kep_deg / DEGPERRAD / 2.0)
        
        self.dens_cb = params.dens_cb
        self.dens_prof_ire = params.density_profile_ire
        self.dens_prof_kep = params.density_profile_kep
        
        self.temp_cb = params.temp_cb
        self.temp_prof_ire = params.temp_profile_ire
        self.temp_prof_kep = params.temp_profile_kep
        
        self.velrot_cb_x_rcb = np.sqrt(2.0 * self.mass_grav * self.r_cb) * self.rot
        
        # State holders
        self.mesh = None
        self.sky = None
        self.pv_diag = None

    def calculate_physics(self, r_cyl, theta, z):
        """
        Calculate velocity, density, and temperature fields.
        """
        dist_protostar = np.sqrt(r_cyl**2 + z**2)
        
        # Determine regions
        region_id = np.full_like(r_cyl, NO_GAS, dtype=int)
        
        # Distance bounds
        valid_dist = (dist_protostar <= self.r_out) & (dist_protostar >= self.r_in) & (dist_protostar > EPS)
        
        is_ire_cond = (dist_protostar >= self.r_cb) & \
                      (np.abs(z) < r_cyl * self.tan_flare_ire + self.height_ire / 2.0)
                      
        is_kep_cond = (dist_protostar < self.r_cb) & \
                      (np.abs(z) < r_cyl * self.tan_flare_kep + self.height_kep / 2.0)
        
        region_id[valid_dist & is_ire_cond] = IS_IRE
        region_id[valid_dist & is_kep_cond] = IS_KEP
        
        # Initialize outputs
        vr = np.zeros_like(r_cyl)
        vtheta = np.zeros_like(r_cyl)
        vz = np.zeros_like(r_cyl)
        density = np.zeros_like(r_cyl)
        temperature = np.zeros_like(r_cyl)
        
        # Avoid division by zero in physics calc
        safe_dist = np.maximum(dist_protostar, EPS)
        
        # --- KEP Calculation ---
        mask_kep = (region_id == IS_KEP)
        if np.any(mask_kep):
            # Velocity
            vtheta[mask_kep] = np.sqrt(self.mass_grav / safe_dist[mask_kep]) * self.rot
            # vr and vz remain 0
            
            # Density & Temperature
            ratio = safe_dist[mask_kep] / self.r_cb
            density[mask_kep] = self.dens_cb * np.power(ratio, self.dens_prof_kep)
            temperature[mask_kep] = self.temp_cb * np.power(ratio, self.temp_prof_kep)
            
        # --- IRE Calculation ---
        mask_ire = (region_id == IS_IRE)
        if np.any(mask_ire):
            dist_ire = safe_dist[mask_ire]
            r_ire = r_cyl[mask_ire]
            z_ire = z[mask_ire]
            
            # Velocity
            vt_val = self.velrot_cb_x_rcb / dist_ire
            vtheta[mask_ire] = vt_val
            
            vel_inf2 = 2.0 * self.mass_grav / dist_ire - np.square(vt_val)
            vel_inf = np.sqrt(np.maximum(vel_inf2, 0.0))
            
            vr[mask_ire] = -vel_inf * r_ire / dist_ire
            vz[mask_ire] = -vel_inf * z_ire / dist_ire
            
            # Density & Temperature
            ratio = dist_ire / self.r_cb
            density[mask_ire] = self.dens_cb * np.power(ratio, self.dens_prof_ire)
            temperature[mask_ire] = self.temp_cb * np.power(ratio, self.temp_prof_ire)
            
        return vr, vtheta, vz, density, temperature

    def make_cube(self, output_filename=None):
        """
        Create the full 3D model, project to sky, convolve, and optionally save to FITS.
        """
        # Mesh setup
        mesh_crpix = [self.npix/2 - 1] * 3
        mesh_crval = [0.0] * 3
        # FIXED: Use AU for mesh
        cdelt_au = self.params.fldres_as * self.params.distance_pc
        mesh_cdelt = [cdelt_au] * 3
        
        self.mesh = Mesh(self.npix, mesh_crpix, mesh_crval, mesh_cdelt, 
                         self.params.inc_deg, self.params.pa_deg)
        
        r, theta, z = self.mesh.get_pos_polar()
        vr, vtheta, vz, dens, temp = self.calculate_physics(r, theta, z)
        
        self.mesh.set_vel_polar(vr, vtheta, vz)
        self.mesh.set_scalar_data(dens, temp)
        
        sky_crpix = [self.npix/2 - 1, self.npix/2 - 1, self.nvel/2 - 1]
        sky_crval = [0.0, 0.0, 0.0]
        # FIXED: Use CM for sky projection
        cdelt_cm = cdelt_au * CMPERAU
        sky_cdelt = [
            -cdelt_cm, # RA increases to left
            cdelt_cm, 
            self.params.velres_kmps * CMPERKM
        ]
        
        self.sky = SkyPlane(self.npix, self.nvel, sky_crpix, sky_crval, sky_cdelt,
                            self.params.linewidth_kmps, self.params.beam_maj_as, 
                            self.params.beam_min_as, self.params.beam_pa_deg,
                            self.params.distance_pc)
        
        self.sky.projection(self.mesh.data)
        self.sky.convolve()
        
        if output_filename:
            print(f"Writing Cube: {output_filename}")
            write_fits_cube(output_filename, self.sky, self.params)
            
        return self.sky

    def make_moments(self, output_prefix=None, moment_type='all', threshold=0.01, plot=True):
        """
        Generate Moment-0 and Moment-1 maps.
        
        Args:
            output_prefix (str): Prefix for output FITS files (e.g. "model" -> "model_mom0.fits")
            moment_type (str): 'mom0', 'mom1', or 'all' (default).
            threshold (float): Relative intensity threshold (0.0-1.0) for masking.
            plot (bool): Whether to generate plots (PNG).
        """
        if self.sky is None:
            raise RuntimeError("SkyPlane not initialized. Run make_cube() first.")
            
        mom0, mom1 = self.sky.calculate_moments(threshold=threshold)
        
        if output_prefix:
            if moment_type in ['mom0', 'all']:
                fname = f"{output_prefix}_mom0.fits"
                print(f"Writing Mom0: {fname}")
                write_fits_moment(fname, mom0, self.sky, self.params, moment=0)
                if plot:
                    plot_moment(fname, 'mom0', f"{output_prefix}_mom0.png")
                
            if moment_type in ['mom1', 'all']:
                fname = f"{output_prefix}_mom1.fits"
                print(f"Writing Mom1: {fname}")
                write_fits_moment(fname, mom1, self.sky, self.params, moment=1)
                if plot:
                    plot_moment(fname, 'mom1', f"{output_prefix}_mom1.png")
                
        return mom0, mom1

    def make_pv(self, pv_pa_deg: float = None, pv_off_ra: float = None, pv_off_dec: float = None, output_filename: str = "example_pv.fits", plot=True):
        """
        Generate PV diagram from the SkyPlane.
        """
        if self.sky is None:
            raise RuntimeError("SkyPlane not initialized. Run make_cube() first.")
            
        if pv_pa_deg is None: pv_pa_deg = 0.0
        if pv_off_ra is None: pv_off_ra = 0.0
        if pv_off_dec is None: pv_off_dec = 0.0
        
        print("Generating PV Diagram...")
        self.pv_diag = PVDiagram(self.npix, self.params, pv_pa_deg, pv_off_ra, pv_off_dec)
        self.pv_diag.generate(self.sky, self.params)
        
        if output_filename:
            print(f"Writing PV: {output_filename}")
            write_fits_pv(output_filename, self.pv_diag.data, self.pv_diag, self.params)
            if plot:
                plot_pv(output_filename, output_filename.replace('.fits', '.png'))
            
        return self.pv_diag

def generate_grid(params_dict, output_dir="grid_output", prefix="model", npix=128, nvel=64):
    """
    Generate a grid of models based on input parameters.
    If any parameter in params_dict is iterable (and not a string), it is treated as a grid axis.
    
    Args:
        params_dict (dict): Dictionary of parameters for SourceParams.
                            Values can be scalars or lists/arrays.
        output_dir (str): Directory to save output files.
        prefix (str): Prefix for output filenames.
        npix (int): Grid size (spatial).
        nvel (int): Grid size (velocity).
    """
    import matplotlib.pyplot as plt
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    # Identify varying parameters
    varying_keys = []
    varying_values = []
    fixed_params = {}
    
    for k, v in params_dict.items():
        # Treat strings as scalars (even though they are iterable)
        if isinstance(v, (list, np.ndarray, tuple)) and not isinstance(v, str):
             varying_keys.append(k)
             varying_values.append(v)
        else:
             fixed_params[k] = v
             
    if not varying_keys:
        print("No varying parameters found. Running single model.")
        # Wrap single run logic if needed, or just warn
        # For consistency, we treat it as 1 combination
        combinations = [()]
    else:
        # Cartesian product of all varying parameters
        combinations = list(itertools.product(*varying_values))
        print(f"Found {len(varying_keys)} varying parameters: {varying_keys}")
        print(f"Total models to generate: {len(combinations)}")

    # Loop through combinations
    for idx, combo in enumerate(combinations):
        current_params = fixed_params.copy()
        
        # Build filename suffix
        suffix_parts = []
        
        for i, key in enumerate(varying_keys):
            val = combo[i]
            current_params[key] = val
            
            # Format value for filename
            if isinstance(val, float):
                # Avoid decimal points in filenames if possible, or use 'p'
                # e.g. 0.5 -> 0p5
                val_str = f"{val:.2g}".replace('.', 'p')
            else:
                val_str = str(val)
            
            # Shorten key names if possible? User asked for "mass_xxx_rcb_yyy"
            # We use full key name or simple mapping? 
            # User example: mass_msun -> mass
            # Let's just use the key name to be safe and generic
            suffix_parts.append(f"{key}_{val_str}")
            
        if suffix_parts:
            suffix = "_" + "_".join(suffix_parts)
        else:
            suffix = ""
            
        full_prefix = f"{output_dir}/{prefix}{suffix}"
        fits_name = f"{full_prefix}.fits"
        current_params['outputfilename'] = fits_name
        
        print(f"\n[{idx+1}/{len(combinations)}] Generating model: {fits_name}")
        
        try:
            # Create SourceParams object
            # Filter out keys that are not in SourceParams (if any extra keys are in dict)
            # SourceParams is a dataclass, strict init by default?
            # We assume params_dict keys match SourceParams fields
            source_params = SourceParams(**current_params)
        except TypeError as e:
            print(f"Error creating SourceParams for {suffix}: {e}")
            continue
            
        # Initialize and Run Model
        try:
            model = IREModel(source_params, npix=npix, nvel=nvel)
            
            # Make Cube
            model.make_cube(output_filename=fits_name)
            
            # Make PV (Default parameters)
            # Use sensible defaults for PV cut (e.g. along major axis)
            # Or use fixed_params if pv_pa_deg is provided there?
            # SourceParams has beam_pa_deg but not pv_pa_deg.
            # We default to PA=0 or user needs to provide PV params?
            # Let's use 0,0,0 or maybe better: Along the PA of the source?
            # source.pa_deg is the elongation. Usually we cut along major axis.
            # Let's use pv_pa_deg = source_params.pa_deg
            
            pv_pa = source_params.pa_deg
            model.make_pv(pv_pa_deg=pv_pa, pv_off_ra=0.0, pv_off_dec=0.0, 
                          output_filename=f"{full_prefix}_PV.fits", plot=True)
            
            # Make Moments
            model.make_moments(output_prefix=full_prefix, moment_type='all', plot=True)
            
            # Close plots to free memory
            plt.close('all')
            
        except Exception as e:
            print(f"Failed to generate model {suffix}: {e}")
            import traceback
            traceback.print_exc()

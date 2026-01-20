import numpy as np
from fitter import fit_grid
from params import SourceParams
from models import IREModel, generate_grid

def generate_true_observation(true_params_dict):
    """
    Generate the 'True' observation with specific parameters.
    """
    params = SourceParams(**true_params_dict)
    print(f"Generating True Observation (Mass={params.mass_msun}, Rout={params.rout_au})...")
    
    model = IREModel(params, npix=128, nvel=64)
    model.make_cube(output_filename="true_obs.fits")
    model.make_moments(output_prefix="true_obs", moment_type='mom1', plot=True)
    return "true_obs_mom1.fits"

def main():
    # 1. Define Base Parameters (from example.py)
    base_params = dict(
        outputfilename="dummy.fits",
        fldres_as=0.05,
        velres_kmps=0.686,
        distance_pc=5100.0,
        beam_maj_as=0.37451,
        beam_min_as=0.29969,
        beam_pa_deg=54.3281,
        linewidth_kmps=1.0,
        
        # Default Physics (will be overridden)
        mass_msun=450.0,
        rcb_au=2000.0,
        
        inc_deg=-45.0,
        pa_deg=160.0,
        rot_sign=1.0,
        rout_au=15000.0,
        rin_au=1000.0,
        
        height_ire_au=0.0,
        flare_ire_deg=30.0,
        density_profile_ire=-1.5,
        temp_profile_ire=0.0,
        height_kep_au=0.0,
        flare_kep_deg=30.0,
        density_profile_kep=-1.5,
        temp_profile_kep=0.0,
        dens_cb=1.0e-2,
        temp_cb=10.0,
        
        name_line="CH3OH",
        restfreq_ghz=1.0,
        name_object="Fit_Test",
        radesys="ICRS",
        center_ra_str="16h09m52.5704s",
        center_dec_str="-51d54m54.3644s",
        vsys_kmps=0.0
    )

    # 2. Generate "True" Observation
    # Target: Mass=450 Msun, Rcb=4000 AU, Rout=15000 AU
    true_params = base_params.copy()
    true_params['mass_msun'] = 450.0
    true_params['rcb_au'] = 4000.0
    true_params['rout_au'] = 15000.0
    
    obs_file = generate_true_observation(true_params)
    
    # 3. Define Grid for Search
    # Mass: 400-500, step 10 -> np.arange(400, 501, 10)
    # Rcb: 3500-4500, step 100 -> np.arange(3500, 4501, 100)
    
    grid = {
        'mass_msun': np.arange(400, 501, 10),
        'rcb_au': np.arange(3500, 4501, 100)
    }
    
    # 4. Run Fitting
    print("\n--- Starting Grid Fit ---")
    # Use base_params as template (values not in grid will be kept fixed)
    # Ensure template has correct fixed Rout
    base_params['rout_au'] = 15000.0
    
    fit_grid(
        obs_file=obs_file,
        template_params=base_params,
        param_grid=grid,
        output_dir="fit_results_demo_rcb",
        npix=128,
        nvel=64
    )
    
    print("Done! Check fit_results_demo_rcb/ for plots.")

if __name__ == "__main__":
    main()

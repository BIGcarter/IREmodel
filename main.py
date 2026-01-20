import sys
import argparse
import numpy as np
import time
from config import CMPERAU, CMPERPC, CMPERKM, VAXIS, NAXIS, NAXIS_PV
from params import SourceParams
from mesh import Mesh
from models import IREModel
from sky import SkyPlane
from pv_diagram import PVDiagram
from io_utils import write_fits_cube, write_fits_pv

# Constants for grid size (from feria.h)
LB_NPIX = 7
LB_NVEL = 6
NPIX = 1 << LB_NPIX
NVEL = 1 << LB_NVEL

def parse_input_file(filepath):
    """
    Parse the input file strictly following the C++ reading order.
    """
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
            
    # Sequential iterator
    it = iter(lines)
    
    try:
        # 1. Output filename
        filename_cubefits = next(it).split()[0]
        
        # 2. Overwrite flag
        dummy_char = next(it).split()[0]
        f_overwrite = (dummy_char == 'y')
        
        # 3. Object Name
        name_object = next(it).split()[0]
        
        # 4. RADESYS
        radesys = next(it).split()[0]
        
        # 5. Center RA
        center_ra = next(it).split()[0]
        
        # 6. Center Dec
        center_dec = next(it).split()[0]
        
        # 7. Vsys
        vsys_kmps = float(next(it).split()[0])
        
        # 8. Line Name
        name_line = next(it).split()[0]
        
        # 9. Rest Freq
        restfreq_ghz = float(next(it).split()[0])
        
        # 10. Fld Res
        fldres_as = float(next(it).split()[0])
        
        # 11. Vel Res
        velres_kmps = float(next(it).split()[0])
        
        # 12. Distance
        distance_pc = float(next(it).split()[0])
        
        # 13. Mass
        mass_msun = float(next(it).split()[0])
        
        # 14. rCB
        rcb_au = float(next(it).split()[0])
        
        # 15. Inc
        inc_deg = float(next(it).split()[0])
        
        # 16. PA
        pa_deg = float(next(it).split()[0])
        
        # 17. Rot sign
        rot_sign = float(next(it).split()[0])
        
        # 18. Rout
        rout_au = float(next(it).split()[0])
        
        # 19. Rin
        rin_au = float(next(it).split()[0])
        
        # 20. Height IRE
        height_ire_au = float(next(it).split()[0])
        
        # 21. Flare IRE
        flare_ire_deg = float(next(it).split()[0])
        
        # 22. Dens Profile IRE
        density_profile_ire = float(next(it).split()[0])
        
        # 23. Temp Profile IRE
        temp_profile_ire = float(next(it).split()[0])
        
        # 24. Height KEP
        height_kep_au = float(next(it).split()[0])
        
        # 25. Flare KEP
        flare_kep_deg = float(next(it).split()[0])
        
        # 26. Dens Profile KEP
        density_profile_kep = float(next(it).split()[0])
        
        # 27. Temp Profile KEP
        temp_profile_kep = float(next(it).split()[0])
        
        # 28. Dens CB
        dens_cb = float(next(it).split()[0])
        
        # 29. Temp CB
        temp_cb = float(next(it).split()[0])
        
        # 30. Linewidth
        linewidth_kmps = float(next(it).split()[0])
        
        # 31. Beam Maj
        beam_maj_as = float(next(it).split()[0])
        
        # 32. Beam Min
        beam_min_as = float(next(it).split()[0])
        
        # 33. Beam PA
        beam_pa_deg = float(next(it).split()[0])
        
        # Adjust beam (C++ logic)
        if beam_maj_as < beam_min_as:
            beam_maj_as, beam_min_as = beam_min_as, beam_maj_as
            beam_pa_deg = 90.0 - beam_pa_deg
            
        # 34. Norm flag
        dummy_char = next(it).split()[0]
        f_norm = (dummy_char == 'y')
        
        # 35. PV PA
        pv_pa_deg = float(next(it).split()[0])
        
        # 36. PV Offset RA
        pv_cent_offset_ra_au = float(next(it).split()[0])
        
        # 37. PV Offset Dec
        pv_cent_offset_dec_au = float(next(it).split()[0])
        
    except StopIteration:
        raise ValueError("Input file ended unexpectedly")
        
    params = SourceParams(
        outputfilename=filename_cubefits,
        fldres_as=fldres_as,
        velres_kmps=velres_kmps,
        distance_pc=distance_pc,
        mass_msun=mass_msun,
        rcb_au=rcb_au,
        inc_deg=inc_deg,
        pa_deg=pa_deg,
        rot_sign=rot_sign,
        rout_au=rout_au,
        rin_au=rin_au,
        height_ire_au=height_ire_au,
        flare_ire_deg=flare_ire_deg,
        height_kep_au=height_kep_au,
        flare_kep_deg=flare_kep_deg,
        dens_cb=dens_cb,
        density_profile_ire=density_profile_ire,
        density_profile_kep=density_profile_kep,
        temp_cb=temp_cb,
        temp_profile_ire=temp_profile_ire,
        temp_profile_kep=temp_profile_kep,
        linewidth_kmps=linewidth_kmps,
        beam_maj_as=beam_maj_as,
        beam_min_as=beam_min_as,
        beam_pa_deg=beam_pa_deg,
        name_line=name_line,
        restfreq_ghz=restfreq_ghz,
        name_object=name_object,
        radesys=radesys,
        center_ra_str=center_ra,
        center_dec_str=center_dec,
        vsys_kmps=vsys_kmps
    )
    
    return params, f_overwrite, f_norm, pv_pa_deg, pv_cent_offset_ra_au, pv_cent_offset_dec_au

def main():
    parser = argparse.ArgumentParser(description="FERIA: IRE Model Generator")
    parser.add_argument("input_file", help="Path to input parameter file")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"Reading input file: {args.input_file}")
    try:
        params, f_overwrite, f_norm, pv_pa_deg, pv_off_ra, pv_off_dec = parse_input_file(args.input_file)
    except Exception as e:
        print(f"Error parsing input file: {e}")
        sys.exit(1)
        
    print(f"\nObject: {params.name_object}")
    print(f"Grid Size: {NPIX} x {NPIX} x {NVEL}")
    
    # Initialize Mesh
    # Mesh center is 0, 0, 0
    # C++: meshCrpix = npix/2 - 1
    # We want center to be at index (npix-1)/2 if using 0-based indexing for symmetry?
    # C++: meshCrpix[i] = npix / 2. - 1.;
    
    mesh_crpix = [NPIX / 2.0 - 1.0] * 3
    mesh_crval = [0.0] * 3
    # meshCdelt_au = Fldres_as * distance_pc
    cdelt_val = params.fldres_as * params.distance_pc
    mesh_cdelt = [cdelt_val] * 3
    
    print("Initializing Mesh...")
    mesh = Mesh(NPIX, mesh_crpix, mesh_crval, mesh_cdelt, params.inc_deg, params.pa_deg)
    
    print("Initializing Physics Model...")
    model = IREModel(params)
    
    print("Calculating Physics...")
    # Get polar coordinates
    r, theta, z = mesh.get_pos_polar()
    
    # Calculate physics
    vr, vtheta, vz, dens, temp = model.calculate_physics(r, theta, z)
    
    # Set data to mesh
    mesh.set_vel_polar(vr, vtheta, vz)
    mesh.set_scalar_data(dens, temp)
    
    print("Projecting to Sky Plane...")
    # Initialize SkyPlane
    # skyCrpix[i] = (i == VAXIS ? nvel / 2 - 1: npix / 2 - 1);
    sky_crpix = [
        NPIX / 2 - 1,   # x
        NPIX / 2 - 1,   # y
        NVEL / 2 - 1    # v
    ]
    sky_crval = [0.0, 0.0, 0.0]
    # skyCdelt[i] = (i == VAXIS ? Velres_kmps : Fldres_as * distance_pc * pow(-1, i + 1));
    # Note: C++ uses pow(-1, i+1) for spatial axes. 
    # i=0 (X): -1^1 = -1. Wait. i=0 is X. i+1=1. pow(-1, 1) = -1.
    # i=1 (Y): -1^2 = 1.
    # Typically RA decreases to the right (positive X).
    # If CDELT1 < 0, RA increases to left.
    # Let's check C++ config.
    # In C++: XAXIS=0, YAXIS=1.
    # skyCdelt[0] = Fldres * dist * (-1). Negative.
    # skyCdelt[1] = Fldres * dist * (1). Positive.
    
    sky_cdelt = [
        params.fldres_as * params.distance_pc * CMPERAU * -1.0,
        params.fldres_as * params.distance_pc * CMPERAU * 1.0,
        params.velres_kmps * CMPERKM # Velocity in cm/s
    ]
    
    sky = SkyPlane(NPIX, NVEL, sky_crpix, sky_crval, sky_cdelt,
                   params.linewidth_kmps, params.beam_maj_as, params.beam_min_as, params.beam_pa_deg,
                   params.distance_pc)
                   
    sky.projection(mesh.data)
    
    print("Convolving...")
    sky.convolve()
    
    if f_norm:
        print("Normalizing...")
        sky.normalize()
        
    print(f"Writing Cube: {params.outputfilename}")
    # Note: C++ dynamically builds filename based on parameters if needed.
    # Python version uses filename from input for now.
    write_fits_cube(params.outputfilename, sky, params, overwrite=f_overwrite)
    
    print("Generating PV Diagram...")
    pv = PVDiagram(NPIX, params, pv_pa_deg, pv_off_ra, pv_off_dec)
    pv.generate(sky, params)
    
    pv_filename = params.outputfilename.replace(".fits", "") + "_PV.fits"
    print(f"Writing PV: {pv_filename}")
    write_fits_pv(pv_filename, pv.data, pv, params, overwrite=f_overwrite)
    
    elapsed = time.time() - start_time
    print(f"\nDone. Duration: {elapsed:.2f} sec")

if __name__ == "__main__":
    main()

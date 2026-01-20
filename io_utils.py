from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
import numpy as np
import os
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)
from config import CMPERKM, CMPERAU, EQUINOX, FITS_EXT, XAXIS, YAXIS, VAXIS
from params import SourceParams
from sky import SkyPlane
from coords import arcsec2radec
import numpy as np

def write_fits_cube(filename: str, sky: SkyPlane, source: SourceParams, overwrite=True):
    """
    Write the 3D data cube to a FITS file.
    """
    if not filename.endswith(FITS_EXT):
        filename += FITS_EXT
        
    # Prepare data: Transpose to (nvel, ny, nx) for FITS standard (v, y, x)
    # Sky.emission is (nx, ny, nvel)
    data = np.transpose(sky.emission, (2, 1, 0))
    
    # FERIA C++ behavior match:
    # Mesh coordinates are typically West -> East (increasing index)
    # FITS coordinates are East -> West (increasing index, negative CDELT)
    # So we need to flip the X-axis (RA) to map Mesh East to FITS East.
    # C++ does: emission_fits[...] = emission[npix - 1 - ira]...
    # which effectively reverses the X-axis.
    data = data[:, :, ::-1]
    
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    
    # Standard Keys & Comments
    header['BSCALE'] = (1.0, 'PHYSICAL = PIXEL*BSCALE + BZERO')
    header['BZERO'] = (0.0, '')
    header['BUNIT'] = ('JY/BEAM', 'Brightness (pixel) unit')
    header['BTYPE'] = 'Intensity'
    header['OBJECT'] = (source.name_object, f'for {source.name_object}')
    header['EQUINOX'] = (2000.0, '')
    header['RADESYS'] = (source.radesys, '')
    header['SPECSYS'] = ('LSRK', 'Spectral reference frame')
    
    # Beam Info
    header['BMAJ'] = (sky.beam_maj_as / 3600., f'{sky.beam_maj_as} as')
    header['BMIN'] = (sky.beam_min_as / 3600., f'{sky.beam_min_as} as')
    header['BPA'] = (sky.beam_pa_deg, 'deg')
    
    # WCS
    # Axis 1: RA (nx)
    header['CTYPE1'] = 'RA---SIN'
    header['CRVAL1'] = (source.cent_ra_deg, source.center_ra_str)
    header['CDELT1'] = (sky.cdelt[XAXIS] / CMPERAU / source.distance_pc / 3600., f'{sky.cdelt[XAXIS] / CMPERAU} au')
    header['CRPIX1'] = sky.crpix[XAXIS] + 1
    header['CUNIT1'] = 'deg'
    
    # Axis 2: Dec (ny)
    header['CTYPE2'] = 'DEC--SIN'
    header['CRVAL2'] = (source.cent_dec_deg, source.center_dec_str)
    header['CDELT2'] = (sky.cdelt[YAXIS] / CMPERAU / source.distance_pc / 3600., f'{sky.cdelt[YAXIS] / CMPERAU} au')
    header['CRPIX2'] = sky.crpix[YAXIS] + 1
    header['CUNIT2'] = 'deg'
    
    # Axis 3: Velocity (nvel)
    header['CTYPE3'] = 'VRAD'
    header['CRVAL3'] = (source.vsys_kmps * 1000.0 / 100.0, 'm/s') # Vsys in cm/s -> m/s
    header['CDELT3'] = (sky.cdelt[VAXIS] / 100.0, 'm/s') # cm/s -> m/s
    header['CRPIX3'] = sky.crpix[VAXIS] + 1
    header['CUNIT3'] = 'm/s'
    
    header['RESTFREQ'] = (source.restfreq_hz, f'Rest Frequency (Hz) of {source.name_line}')
    
    # Comments
    header.add_comment("This is a result of an infalling-rotating envelope model with the following physical parameters")
    
    # Custom IRE Parameters
    import warnings
    from astropy.io.fits.verify import VerifyWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=VerifyWarning)
        _add_ire_keys(header, source)
        hdu.writeto(filename, overwrite=overwrite, output_verify='silentfix')
    
    print(f"Fits file was created: {filename}")

def write_fits_pv(filename: str, pv_data, pv_diag, source: SourceParams, overwrite=True):
    """
    Write the PV diagram to a FITS file.
    """
    try:
        from astropy.io import fits
        from astropy.io.fits.verify import VerifyWarning
    except Exception as e:
        print(f"Warning: Could not import astropy.io.fits (Environment Issue): {e}")
        return

    if not filename.endswith(FITS_EXT):
        filename += FITS_EXT
        
    # PV Data shape: (npix_pv, nvel) -> FITS (nvel, npix_pv)
    data = np.transpose(pv_data, (1, 0))
    
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    
    # Standard Keys & Comments
    header['BSCALE'] = (1.0, 'PHYSICAL = PIXEL*BSCALE + BZERO')
    header['BZERO'] = (0.0, '')
    header['BUNIT'] = ('JY/BEAM', 'Brightness (pixel) unit')
    header['BTYPE'] = 'Intensity'
    header['OBJECT'] = (f"PV of IRE model for {source.name_object}", f'for {source.name_object}')
    header['EQUINOX'] = (2000.0, '')
    header['RADESYS'] = (source.radesys, '')
    header['SPECSYS'] = ('LSRK', 'Spectral reference frame')
    
    # Beam Info
    header['BMAJ'] = (pv_diag.beam_maj_as / 3600., f'{pv_diag.beam_maj_as} as')
    header['BMIN'] = (pv_diag.beam_min_as / 3600., f'{pv_diag.beam_min_as} as')
    header['BPA'] = (pv_diag.beam_pa_deg, 'deg')
    
    # WCS
    # Axis 1: Offset (deg)
    header['CTYPE1'] = 'ANGLE'
    # pv_diag.crval is in cm, convert to deg
    # PV CRVAL is 0 (offset)
    header['CRVAL1'] = 0.0
    header['CDELT1'] = (pv_diag.cdelt_au / source.distance_pc / 3600., f'{pv_diag.cdelt_au} au')
    header['CRPIX1'] = pv_diag.npix_pv / 2.0 + 0.5 # Center pixel
    header['CUNIT1'] = 'deg'
    
    # Axis 2: Velocity
    header['CTYPE2'] = 'VRAD'
    header['CRVAL2'] = (source.vsys_kmps * 1000.0, 'm/s') # km/s -> m/s
    header['CDELT2'] = (pv_diag.velres_kmps * 1000.0, 'm/s')
    header['CRPIX2'] = pv_diag.nvel / 2.0 + 0.5
    header['CUNIT2'] = 'm/s'
    
    header['RESTFREQ'] = (source.restfreq_hz, f'Rest Frequency (Hz) of {source.name_line}')
    
    # Comments
    header.add_comment("This is a result of an infalling-rotating envelope model with the following physical parameters")
    _add_ire_keys(header, source)
    
    header.add_comment("This is a result of a PV slice of the above IRE model with the following parameters")
    _add_pv_keys(header, pv_diag, source)
    
    hdu.writeto(filename, overwrite=overwrite, output_verify='silentfix')
    print(f"Fits file was created: {filename}")

def write_fits_moment(filename: str, moment_data, sky: SkyPlane, source: SourceParams, moment=0, overwrite=True):
    """
    Write Moment-0 or Moment-1 map to FITS.
    """
    try:
        from astropy.io import fits
        from astropy.io.fits.verify import VerifyWarning
    except Exception as e:
        print(f"Warning: Could not import astropy.io.fits (Environment Issue): {e}")
        return

    if not filename.endswith(FITS_EXT):
        filename += FITS_EXT
        
    # Moment data is (nx, ny). Transpose to (ny, nx) for FITS.
    data = moment_data.T
    
    # Flip X-axis (RA) to match C++ FERIA behavior (East on left)
    data = data[:, ::-1]
    
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    
    # Standard Keys
    header['BSCALE'] = (1.0, 'PHYSICAL = PIXEL*BSCALE + BZERO')
    header['BZERO'] = (0.0, '')
    
    if moment == 0:
        header['BUNIT'] = ('JY/BEAM.KM/S', 'Integrated Intensity')
        header['BTYPE'] = 'Moment-0'
    elif moment == 1:
        header['BUNIT'] = ('KM/S', 'Velocity Field')
        header['BTYPE'] = 'Moment-1'
        
    header['OBJECT'] = (source.name_object, f'for {source.name_object}')
    header['EQUINOX'] = (2000.0, '')
    header['RADESYS'] = (source.radesys, '')
    
    # Beam Info
    header['BMAJ'] = (sky.beam_maj_as / 3600., f'{sky.beam_maj_as} as')
    header['BMIN'] = (sky.beam_min_as / 3600., f'{sky.beam_min_as} as')
    header['BPA'] = (sky.beam_pa_deg, 'deg')
    
    # WCS
    header['CTYPE1'] = 'RA---SIN'
    header['CRVAL1'] = (source.cent_ra_deg, source.center_ra_str)
    header['CDELT1'] = (sky.cdelt[XAXIS] / CMPERAU / source.distance_pc / 3600., f'{sky.cdelt[XAXIS] / CMPERAU} au')
    header['CRPIX1'] = sky.crpix[XAXIS] + 1
    header['CUNIT1'] = 'deg'
    
    header['CTYPE2'] = 'DEC--SIN'
    header['CRVAL2'] = (source.cent_dec_deg, source.center_dec_str)
    header['CDELT2'] = (sky.cdelt[YAXIS] / CMPERAU / source.distance_pc / 3600., f'{sky.cdelt[YAXIS] / CMPERAU} au')
    header['CRPIX2'] = sky.crpix[YAXIS] + 1
    header['CUNIT2'] = 'deg'
    
    header['RESTFREQ'] = (source.restfreq_hz, f'Rest Frequency (Hz) of {source.name_line}')
    
    # Comments
    header.add_comment(f"This is a Moment-{moment} map of the IRE model")
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=VerifyWarning)
        _add_ire_keys(header, source)
        hdu.writeto(filename, overwrite=overwrite, output_verify='silentfix')
    
    print(f"Fits file was created: {filename}")

def _add_ire_keys(header, source: SourceParams):
    # Match keys and comments from feria_fitsio.cpp
    header['IREOBJ'] = (source.name_object, 'Target source')
    header['IRELINE'] = (source.name_line, 'Molecular line')
    header['IREDIST'] = (source.distance_pc, 'Distance (pc)')
    header['IREMASS'] = (source.mass_msun, 'Protostellar mass (Msun)')
    header['IRERCB'] = (source.rcb_au, 'Radius of the centrifugal barrier (au)')
    header['IREINC'] = (source.inc_deg, 'Inclination angle (0 degree for face-on; deg)')
    header['IREPA'] = (source.pa_deg, 'Position angle of the elongation (deg)')
    header['IREROT'] = (source.rot_sign, 'Rotation (\'1\' for positive)')
    header['IREROUT'] = (source.rout_au, 'Outer radius (au)')
    header['IRERIN'] = (source.rin_au, 'Inner radius (au)')
    
    header['IRETIRE'] = (source.height_ire_au, 'Scale height (au) (IRE)')
    header['IREFLAREIRE'] = (source.flare_ire_deg, 'Flared angle (deg) (IRE)')
    header['IREDPROIRE'] = (source.density_profile_ire, 'Density profile (IRE)')
    header['IRETPROIRE'] = (source.temp_profile_ire, 'Temperature profile (IRE)')
    
    header['IRETKEP'] = (source.height_kep_au, 'Scale height (au) (Kep)')
    header['IREFLAREKEP'] = (source.flare_kep_deg, 'Flared angle (deg) (Kep)')
    header['IREDPROKEP'] = (source.density_profile_kep, 'Density profile (Kep)')
    header['IRETPROKEP'] = (source.temp_profile_kep, 'Temperature profile (Kep)')
    
    header['IREDENS'] = (source.dens_cb, 'Molecular abundance at the CB (cm-3)')
    header['IRETEMP'] = (source.temp_cb, 'Gas kinetic temperature at the CB (K)')
    header['IRELW'] = (source.linewidth_kmps, 'Intrinsic linewidth (km s-1)')
    header['IREBMAJ'] = (source.beam_maj_as, 'Beam (major) (arcsec)')
    header['IREBMIN'] = (source.beam_min_as, 'Beam (minor) (arcsec)')
    header['IREBPA'] = (source.beam_pa_deg, 'Beam (PA) (deg)')
    header['IREFREST'] = (source.restfreq_hz, 'Rest frequency (Hz)')
    header['IREVSYS'] = (source.vsys_kmps, 'Systemic velocity (km s-1)')

def _add_pv_keys(header, pv_diag, source: SourceParams):
    # Calculate formatted coordinates
    # centpv_ra_scaled_as -> HMS
    # centpv_dec_as -> DMS
    
    ra_hms, dec_dms = arcsec2radec(pv_diag.centpv_ra_scaled_as, pv_diag.centpv_dec_as)
    
    centpv_ra_str = f"{ra_hms[0]:02.0f}h{ra_hms[1]:02.0f}m{ra_hms[2]:09.6f}s"
    centpv_dec_str = f"{dec_dms[0]:02.0f}d{dec_dms[1]:02.0f}m{dec_dms[2]:09.6f}s"
    
    header['PVCENTRA'] = (centpv_ra_str, 'Slice Center in RA')
    header['PVCENTDEC'] = (centpv_dec_str, 'Slice Center in DEC')
    header['PVFLDRA'] = (source.center_ra_str, 'Center of Cube in RA')
    header['PVFLDDEC'] = (source.center_dec_str, 'Center of Cube in DEC')
    header['PVOFFRA'] = (pv_diag.cent_offset_ra_au, 'Slice offset from the cube center (au)')
    header['PVOFFDEC'] = (pv_diag.cent_offset_dec_au, 'Sliece offset from the cube center (au)')
    header['PVBEAM'] = (pv_diag.beam_slice_as, f'Beam size along the slice (deg) = {pv_diag.beam_slice_as} as')

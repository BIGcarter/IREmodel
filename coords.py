from typing import Tuple
import numpy as np
from config import DB_ERRATIO

def radec2arcsec(ra_hms: Tuple[float, float, float], dec_dms: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Convert RA (h, m, s) and Dec (d, m, s) to scaled arcseconds.
    
    Args:
        ra_hms: Tuple of (hour, minute, second)
        dec_dms: Tuple of (degree, minute, second)
        
    Returns:
        tuple: (ra_scaled_as, dec_as)
    """
    dec_sign = 1.0
    d, m, s = dec_dms
    if d < 0:
        d = -d
        dec_sign = -1.0
    elif d == 0 and (m < 0 or (m == 0 and s < 0)):
        # Handle -0 cases if passed as such, though tuple usually has sign on first non-zero
        if m < 0: m = -m
        if s < 0: s = -s
        dec_sign = -1.0
        
    # Note: The C++ code logic for negative input handling was:
    # if (dec_dms[0] < 0) for (int i = 1; i < 3; ++i) dec_dms[i] *= -1.;
    # implying input might have negative first component.
    # Here we assume standard input where sign is carried by degrees.
    
    dec_as = (d * 60. + m) * 60. + s
    dec_as *= dec_sign
    
    ra_scaled_as = ((ra_hms[0] * 60. + ra_hms[1]) * 60. + ra_hms[2]) * 360. / 24.
    
    return ra_scaled_as, dec_as

def arcsec2radec(ra_scaled_as: float, dec_as: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Convert scaled arcseconds to RA (h, m, s) and Dec (d, m, s).
    """
    sign_ra = -1 if ra_scaled_as < 0 else 1
    sign_dec = -1 if dec_as < 0 else 1
    
    ra_scaled_as = abs(ra_scaled_as) * 24. / 360.
    dec_as = abs(dec_as)
    
    ra_h = np.floor(ra_scaled_as * (1. + DB_ERRATIO) / 3600.)
    ra_scaled_as -= ra_h * 3600.
    ra_m = np.floor(ra_scaled_as * (1. + DB_ERRATIO) / 60.)
    ra_scaled_as -= ra_m * 60.
    ra_s = ra_scaled_as
    
    dec_d = np.floor(dec_as * (1. + DB_ERRATIO) / 3600.)
    dec_as -= dec_d * 3600.
    dec_m = np.floor(dec_as * (1. + DB_ERRATIO) / 60.)
    dec_as -= dec_m * 60.
    dec_s = dec_as
    
    ra_h *= sign_ra
    dec_d *= sign_dec
    
    return (ra_h, ra_m, ra_s), (dec_d, dec_m, dec_s)

def radec_shift(ra_scaled_as: float, dec_as: float, ra_shift_as: float, dec_shift_as: float) -> Tuple[float, float]:
    dec_shifted_as = dec_as + dec_shift_as
    ra_scaled_as += ra_shift_as / np.cos(np.pi / 180. * (dec_as + dec_shifted_as) / 2. / 3600.)
    return ra_scaled_as, dec_shifted_as

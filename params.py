from dataclasses import dataclass
import numpy as np
from config import CMPERKM, CMPERAU, ASPERRAD

@dataclass
class SourceParams:
    outputfilename: str
    fldres_as: float
    velres_kmps: float
    distance_pc: float
    mass_msun: float
    rcb_au: float
    inc_deg: float
    pa_deg: float
    rot_sign: float
    rout_au: float
    rin_au: float
    height_ire_au: float
    flare_ire_deg: float
    height_kep_au: float
    flare_kep_deg: float
    dens_cb: float
    density_profile_ire: float
    density_profile_kep: float
    temp_cb: float
    temp_profile_ire: float
    temp_profile_kep: float
    linewidth_kmps: float
    beam_maj_as: float
    beam_min_as: float
    beam_pa_deg: float
    name_line: str
    restfreq_ghz: float
    name_object: str
    radesys: str
    center_ra_str: str
    center_dec_str: str
    vsys_kmps: float

    def __post_init__(self):
        self.linewidth_cmps = self.linewidth_kmps * CMPERKM
        self.restfreq_hz = self.restfreq_ghz * 1e9
        self.vsys_cmps = self.vsys_kmps * CMPERKM
        
        # Parse RA/Dec
        try:
            ra_parts = self.center_ra_str.replace('h', ' ').replace('m', ' ').replace('s', ' ').split()
            self.cent_ra = [float(x) for x in ra_parts]
            if len(self.cent_ra) != 3: raise ValueError
            
            self.cent_ra_deg = ((self.cent_ra[2] / 60. + self.cent_ra[1]) / 60. + self.cent_ra[0]) / 24. * 360.
            
            # Dec Parsing
            dec_str = self.center_dec_str.strip()
            dec_sign = -1.0 if dec_str.startswith('-') else 1.0
            
            dec_parts = dec_str.replace('d', ' ').replace('m', ' ').replace('s', ' ').split()
            self.cent_dec = [float(x) for x in dec_parts]
            if len(self.cent_dec) != 3: raise ValueError

            # Calculate magnitude using absolute values
            d = abs(self.cent_dec[0])
            m = abs(self.cent_dec[1])
            s = abs(self.cent_dec[2])
            
            deg_abs = d + m/60.0 + s/3600.0
            self.cent_dec_deg = dec_sign * deg_abs

            self.cent_ra_scaled_as = self.cent_ra_deg * 3600.
            self.cent_dec_as = self.cent_dec_deg * 3600.

        except Exception as e:
            print(f"Warning: Could not parse RA/Dec strings '{self.center_ra_str}', '{self.center_dec_str}': {e}")
            self.cent_ra = [0, 0, 0]
            self.cent_dec = [0, 0, 0]
            self.cent_ra_deg = 0
            self.cent_dec_deg = 0
            self.cent_ra_scaled_as = 0
            self.cent_dec_as = 0

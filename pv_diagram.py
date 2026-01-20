import numpy as np
from config import DEGPERRAD, PI, ASPERRAD, CMPERAU, DB_ERRATIO
from params import SourceParams
from sky import SkyPlane

class PVDiagram:
    def __init__(self, sky_npix, source: SourceParams, pv_pa_deg, cent_offset_ra_au, cent_offset_dec_au):
        self.pv_pa_deg = pv_pa_deg
        self.pv_pa_rad = pv_pa_deg / DEGPERRAD
        
        self.cent_offset_ra_au = cent_offset_ra_au
        self.cent_offset_dec_au = cent_offset_dec_au
        
        self.beam_maj_as = source.beam_maj_as
        self.beam_min_as = source.beam_min_as
        self.beam_pa_deg = source.beam_pa_deg
        self.beam_pa_rad = source.beam_pa_deg / DEGPERRAD
        
        # Calculate PV center coordinates (scaled arcsec)
        # Matches feria_PV.cpp logic
        self.centcube_ra_scaled_as = source.cent_ra_scaled_as
        self.centcube_dec_as = source.cent_dec_as
        
        self.centpv_dec_as = self.centcube_dec_as + self.cent_offset_dec_au / source.distance_pc
        
        # Mean Dec for cos correction
        mean_dec_rad = np.pi / 180.0 * (self.centcube_dec_as + self.centpv_dec_as) / 2.0 / 3600.0
        self.centpv_ra_scaled_as = self.centcube_ra_scaled_as + \
            (self.cent_offset_ra_au / source.distance_pc) / np.cos(mean_dec_rad)

        # Calculate beam slice size
        theta_rad = self.pv_pa_rad - self.beam_pa_rad
        self.beam_slice_as = 1.0 / np.sqrt(
            (np.sin(theta_rad) / self.beam_min_as)**2 + 
            (np.cos(theta_rad) / self.beam_maj_as)**2
        )
        
        # Determine npix for PV
        # npix_PV = ceil(npix / max(abs(cos), abs(sin)))
        cos_pa = np.cos(self.pv_pa_rad)
        sin_pa = np.sin(self.pv_pa_rad)
        factor = max(abs(cos_pa), abs(sin_pa))
        
        self.npix_pv = int(np.ceil(sky_npix / factor * (1.0 - DB_ERRATIO)))
        if self.npix_pv % 2 == 0:
            self.npix_pv += 1
            
        self.nvel = int(1 << 6) # Default, but should come from sky
        
        # Calculate deltas
        # dra_au = Fldres * dist * sin(PA)
        # ddec_au = Fldres * dist * cos(PA) * -1
        # Actually standard projection: dRA ~ sin(PA), dDec ~ cos(PA)
        # Check C++ logic:
        # dra_au = Fldres_as * distance_pc * sin(PV_PA_rad)
        # ddec_au = Fldres_as * distance_pc * cos(PV_PA_rad) * -1.
        
        # We need the step size in pixels along the cut
        # Or step size in AU.
        # Let's derive it from SkyPlane properties when generating data
        self.cdelt_au = 0.0 # To be set
        self.velres_kmps = source.velres_kmps
        
        self.data = None # (npix_pv, nvel)

    def generate(self, sky: SkyPlane, source: SourceParams):
        self.nvel = sky.nvel
        self.data = np.zeros((self.npix_pv, self.nvel))
        
        # Calculate step in AU
        fldres_as = source.fldres_as
        distance_pc = source.distance_pc
        
        dra_au = fldres_as * distance_pc * np.sin(self.pv_pa_rad)
        ddec_au = fldres_as * distance_pc * np.cos(self.pv_pa_rad) * -1.0
        
        step_au = np.sqrt(dra_au**2 + ddec_au**2)
        self.cdelt_au = step_au
        
        # Calculate positions of PV pixels in Sky coordinates
        # Center of PV is 0 offset
        # Sky center is (crpix, crpix)
        
        # Offset from center in AU
        indices = np.arange(self.npix_pv) - (self.npix_pv // 2)
        
        offset_ra_au = -self.cent_offset_ra_au + dra_au * indices
        offset_dec_au = self.cent_offset_dec_au + ddec_au * indices
        
        # Convert AU offsets to Sky Pixel coordinates
        # Sky pixel coords (0-based)
        # index = crpix + (pos - crval) / cdelt
        # pos here is AU. crval is 0. cdelt is AU.
        
        # Sky cdelt in AU
        cdelt_x_au = sky.cdelt[0] / CMPERAU
        cdelt_y_au = sky.cdelt[1] / CMPERAU
        
        x_indices = sky.crpix[0] + offset_ra_au / cdelt_x_au
        y_indices = sky.crpix[1] + offset_dec_au / cdelt_y_au
        
        # Iterate over PV pixels and sample from Sky cube
        # Optimization: use map_coordinates if we want full interpolation
        # But we need to interpolate (x, y) for each velocity channel
        
        # Construct coordinates for map_coordinates
        # coords shape (3, n_samples) -> (x, y, z)
        # We want to sample all v for each (x, y)
        
        from scipy.ndimage import map_coordinates
        
        # We can loop over channels or reshape
        # Let's loop over channels for simplicity, nvel is small (~64)
        
        for iv in range(self.nvel):
            # Coordinates: (x_indices, y_indices)
            # map_coordinates expects (x, y) order matching array dims?
            # sky.emission is (nx, ny, nvel).
            # So dimensions 0 and 1.
            
            coords = np.vstack((x_indices, y_indices))
            
            # Sample
            sampled = map_coordinates(sky.emission[..., iv], coords, order=1, mode='constant', cval=0.0)
            
            self.data[:, iv] = sampled

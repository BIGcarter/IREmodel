import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d
from config import CMPERKM, CMPERAU, ASPERRAD, DEGPERRAD, VAXIS, INF, DB_ERRATIO

class SkyPlane:
    def __init__(self, npix, nvel, sky_crpix, sky_crval, sky_cdelt, 
                 linewidth_kmps, beam_maj_as, beam_min_as, beam_pa_deg, distance_pc):
        
        self.npix = npix
        self.nvel = nvel
        
        # Store WCS info
        self.crpix = np.array(sky_crpix)
        self.crval = np.array(sky_crval)
        self.cdelt = np.array(sky_cdelt)
        
        # Convert to physical units for internal calculation if needed
        # In C++, crval/cdelt are converted to cm (spatial) and cm/s (velocity)
        # Here we keep them in provided units (usually cm for spatial, cm/s for vel if consistent with mesh)
        # But wait, main.cpp sets skyCdelt using Fldres_as * distance_pc.
        # So inputs are likely in cm (spatial) and cm/s (velocity).
        
        self.linewidth_fwhm_cmps = linewidth_kmps * CMPERKM
        
        self.beam_maj_as = beam_maj_as
        self.beam_min_as = beam_min_as
        self.beam_pa_deg = beam_pa_deg
        
        # Beam in cm
        self.beam_maj_cm = beam_maj_as * distance_pc * CMPERAU
        self.beam_min_cm = beam_min_as * distance_pc * CMPERAU
        self.beam_pa_rad = beam_pa_deg / DEGPERRAD
        
        # Emission cube (nx, ny, nvel)
        # Note: numpy order is typically (z, y, x) or (vel, y, x) for FITS.
        # C++ uses [ix][iy][iv]. Let's stick to (nx, ny, nvel) for consistency with mesh
        # and transpose at the end for FITS.
        self.emission = np.zeros((npix, npix, nvel), dtype=np.float64)
        
    def projection(self, mesh_data):
        """
        Project 3D mesh data onto the sky plane (PPV cube).
        
        Args:
            mesh_data: 4D array (nx, ny, nz, ndata) from Mesh class.
        """
        # Extract necessary fields
        # mesh_data shape: (nx, ny, nz, ndata)
        # We need to integrate along z-axis (index 2 of spatial dims)
        # But wait, the projection is defined by mapping Vz to velocity channel.
        # It's not a spatial integration along Z. It's mapping (x,y,z) -> (x,y,v).
        
        from config import VZ_VAL, N_VAL, T_VAL
        
        vz = mesh_data[..., VZ_VAL]  # (nx, ny, nz)
        density = mesh_data[..., N_VAL]
        temperature = mesh_data[..., T_VAL]
        
        emit = density * temperature
        
        # Calculate velocity index
        # iv = (vel - crval) / cdelt + crpix
        iv_float = (vz - self.crval[VAXIS]) / self.cdelt[VAXIS] + self.crpix[VAXIS]
        
        # Splatting logic
        iv_low = np.floor(iv_float).astype(int)
        iv_upp = iv_low + 1
        
        w_upp = iv_float - iv_low
        w_low = 1.0 - w_upp
        
        # Filter bounds
        valid_mask = (iv_low >= -1) & (iv_low < self.nvel) # Relaxed bound to allow partial contribution
        # Actually standard splatting:
        # if iv_low in [0, nvel-1]: add emit * w_low
        # if iv_upp in [0, nvel-1]: add emit * w_upp
        
        # We can use np.add.at
        # Flatten spatial dimensions
        
        # Optimization:
        # Create indices for flat emission array
        nx, ny, nz = vz.shape
        
        # Grid indices for x and y
        ix_grid, iy_grid, _ = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        
        # Flatten everything
        ix_flat = ix_grid.ravel()
        iy_flat = iy_grid.ravel()
        iv_low_flat = iv_low.ravel()
        iv_upp_flat = iv_upp.ravel()
        
        emit_flat = emit.ravel()
        w_low_flat = w_low.ravel()
        w_upp_flat = w_upp.ravel()
        
        # Contribution to low bin
        mask_low = (iv_low_flat >= 0) & (iv_low_flat < self.nvel)
        if np.any(mask_low):
            np.add.at(self.emission, (ix_flat[mask_low], iy_flat[mask_low], iv_low_flat[mask_low]), 
                      emit_flat[mask_low] * w_low_flat[mask_low])
            
        # Contribution to upp bin
        mask_upp = (iv_upp_flat >= 0) & (iv_upp_flat < self.nvel)
        if np.any(mask_upp):
            np.add.at(self.emission, (ix_flat[mask_upp], iy_flat[mask_upp], iv_upp_flat[mask_upp]), 
                      emit_flat[mask_upp] * w_upp_flat[mask_upp])

    def convolve(self):
        """Apply beam and line convolution."""
        self._convolve_line()
        self._convolve_beam()

    def _convolve_line(self):
        # Gaussian kernel for velocity
        # sigma = FWHM / 2.3548
        sigma_v_cmps = self.linewidth_fwhm_cmps / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_pix = sigma_v_cmps / self.cdelt[VAXIS]
        
        # Create 1D kernel
        # Kernel size: usually +/- 3 or 4 sigma
        radius = int(np.ceil(4 * sigma_pix))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma_pix)**2)
        kernel /= kernel.sum()
        
        # Convolve along velocity axis (axis 2)
        self.emission = convolve1d(self.emission, kernel, axis=2, mode='constant', cval=0.0)

    def _convolve_beam(self):
        # 2D Gaussian beam
        # Construct kernel in pixels
        dx_cm = self.cdelt[0] # Assuming square pixels in x/y
        dy_cm = self.cdelt[1]
        
        sigma_maj_cm = self.beam_maj_cm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_min_cm = self.beam_min_cm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        sigma_maj_pix = sigma_maj_cm / dx_cm
        sigma_min_pix = sigma_min_cm / dy_cm
        
        # Rotation
        pa = self.beam_pa_rad
        cpa = np.cos(pa)
        spa = np.sin(pa)
        
        # Kernel grid
        radius = int(np.ceil(4 * max(sigma_maj_pix, sigma_min_pix)))
        x = np.arange(-radius, radius + 1)
        y = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # Rotate coordinates to beam frame
        # x_beam = x cos PA + y sin PA
        # y_beam = -x sin PA + y cos PA
        # Check sign convention. C++ uses:
        # u, v in Fourier space with rotation.
        # Let's use standard Gaussian rotation formula.
        
        xx_rot = xx * cpa + yy * spa
        yy_rot = -xx * spa + yy * cpa
        
        arg = 0.5 * ((xx_rot / sigma_min_pix)**2 + (yy_rot / sigma_maj_pix)**2) # Note: min/maj assignment to axes
        # Usually PA is angle of Major axis. So Major axis lies along X in rotated frame?
        # If PA is angle from North (Y) towards East (X)?
        # C++ code: beam_uv calc.
        # a_u2 involves beam_min_cm * beam_cpa.
        # If u is conjugate to x.
        # Let's trust standard formulation:
        # If PA is angle of major axis from positive Y axis (North) towards positive X (East)?
        # Astronomy convention: PA is East of North.
        # In Cartesian (x=Right/West, y=Up/North): PA is angle from Y towards -X (since RA increases leftwards usually).
        # But here let's assume standard image coords.
        # Let's just use the sigma_maj along the rotated Y axis or X axis depending on PA definition.
        # If PA=0, Major axis is North-South (Y-axis).
        # So aligned with Y.
        # Then sigma_y = sigma_maj, sigma_x = sigma_min.
        
        # Let's assume this alignment.
        arg = 0.5 * ((xx_rot / sigma_min_pix)**2 + (yy_rot / sigma_maj_pix)**2)
        kernel = np.exp(-arg)
        kernel /= kernel.sum()
        
        # Convolve each velocity channel
        # We can use fftconvolve on 2D planes
        # Or 3D convolution with 2D kernel
        
        for iv in range(self.nvel):
            self.emission[..., iv] = fftconvolve(self.emission[..., iv], kernel, mode='same')

    def normalize(self):
        max_val = np.max(self.emission)
        if max_val > 0:
            self.emission /= max_val
            
    def calculate_moments(self, threshold=0.01):
        """
        Calculate Moment-0 (Integrated Intensity) and Moment-1 (Velocity Field).
        
        Args:
            threshold (float): Intensity threshold relative to peak (0.0 to 1.0) for masking.
            
        Returns:
            tuple: (mom0_map, mom1_map)
                   mom0_map: 2D array (nx, ny)
                   mom1_map: 2D array (nx, ny) in km/s (or same units as velocity axis)
        """
        # Emission shape: (nx, ny, nvel)
        # Velocity axis is 2
        
        # Construct velocity array for the spectral axis
        # v = (iv - crpix) * cdelt + crval
        v_axis = (np.arange(self.nvel) - self.crpix[VAXIS]) * self.cdelt[VAXIS] + self.crval[VAXIS]
        
        # Determine mask based on threshold
        max_val = np.max(self.emission)
        mask = self.emission > (threshold * max_val)
        
        # Moment 0: Sum(I * dv)
        # Assuming constant dv (cdelt), we can sum I and multiply by |cdelt|
        dv = np.abs(self.cdelt[VAXIS])
        
        # Apply mask for Mom0? Usually Mom0 includes all, but noise can be an issue.
        # User requested filtering for "mom0, mom1".
        # Let's apply mask to data before summing.
        masked_emission = np.where(mask, self.emission, 0.0)
        
        mom0 = np.sum(masked_emission, axis=2) * dv
        
        # Moment 1: Sum(I * v * dv) / Sum(I * dv)
        
        total_intensity = np.sum(masked_emission, axis=2)
        
        # Handle division by zero (where intensity is 0)
        valid_mask = total_intensity > 0
        
        # Weighted sum of velocities
        # emission * v broadcasted
        weighted_v = np.sum(masked_emission * v_axis[np.newaxis, np.newaxis, :], axis=2)
        
        mom1 = np.zeros_like(total_intensity)
        mom1[valid_mask] = weighted_v[valid_mask] / total_intensity[valid_mask]
        
        # Convert to km/s if needed (Currently in cm/s because SkyPlane works in cm/s internally usually)
        # But wait, in __init__, we set cdelt/crval. 
        # In main.cpp/feria_sky.cpp, VAXIS units are converted to cm/s (CMperKM).
        # So v_axis is in cm/s.
        # We should convert output to km/s for standard FITS usage, or keep it consistent.
        # Let's convert to km/s if the input parameters imply km/s (which they do).
        
        mom1 /= CMPERKM
        
        return mom0, mom1

    def get_val(self, x_idx, y_idx, v_idx):
        """
        Trilinear interpolation to get value at arbitrary index coordinates.
        Using simple implementation similar to C++ getVal.
        """
        # Bounds check
        if (x_idx < 0 or x_idx > self.npix - 1 or 
            y_idx < 0 or y_idx > self.npix - 1 or 
            v_idx < 0 or v_idx > self.nvel - 1):
            return 0.0
            
        # Get integer parts
        x0 = int(np.floor(x_idx))
        y0 = int(np.floor(y_idx))
        v0 = int(np.floor(v_idx))
        
        x1 = min(x0 + 1, self.npix - 1)
        y1 = min(y0 + 1, self.npix - 1)
        v1 = min(v0 + 1, self.nvel - 1)
        
        xd = x_idx - x0
        yd = y_idx - y0
        vd = v_idx - v0
        
        c00 = self.emission[x0, y0, v0] * (1-xd) + self.emission[x1, y0, v0] * xd
        c01 = self.emission[x0, y0, v1] * (1-xd) + self.emission[x1, y0, v1] * xd
        c10 = self.emission[x0, y1, v0] * (1-xd) + self.emission[x1, y1, v0] * xd
        c11 = self.emission[x0, y1, v1] * (1-xd) + self.emission[x1, y1, v1] * xd
        
        c0 = c00 * (1-yd) + c10 * yd
        c1 = c01 * (1-yd) + c11 * yd
        
        c = c0 * (1-vd) + c1 * vd
        return c

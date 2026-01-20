import numpy as np
from config import CMPERAU, DEGPERRAD, NDATA, VX_VAL, VY_VAL, VZ_VAL, N_VAL, T_VAL

class Mesh:
    def __init__(self, npix, mesh_crpix, mesh_crval_au, mesh_cdelt_au, inc_deg, pa_deg):
        self.npix = npix
        
        # Convert to cm
        self.crpix = np.array(mesh_crpix)
        self.crval = np.array(mesh_crval_au) * CMPERAU
        self.cdelt = np.array(mesh_cdelt_au) * CMPERAU
        
        # Adjust angles as in C++ code
        # pa_deg *= -1;
	    # inc_deg -= 90.;
        self.inc_rad = (inc_deg - 90.0) / DEGPERRAD
        self.pa_rad = (-pa_deg) / DEGPERRAD
        
        self.cinc = np.cos(self.inc_rad)
        self.sinc = np.sin(self.inc_rad)
        self.cpa = np.cos(self.pa_rad)
        self.spa = np.sin(self.pa_rad)
        
        # Initialize data array (nx, ny, nz, ndata)
        # Note: C++ loop order ix, iy, iz implies standard C-order if mapped to numpy (ix, iy, iz)
        self.data = np.zeros((npix, npix, npix, NDATA), dtype=np.float64)
        
        # Generate coordinate grids
        # Indices
        i = np.arange(npix)
        j = np.arange(npix)
        k = np.arange(npix)
        
        # Meshgrid for indices (indexing='ij' to match loop order ix, iy, iz)
        self.ix_grid, self.iy_grid, self.iz_grid = np.meshgrid(i, j, k, indexing='ij')
        
        # Cartesian coordinates in mesh frame
        self.x_cart = (self.ix_grid - self.crpix[0]) * self.cdelt[0] + self.crval[0]
        self.y_cart = (self.iy_grid - self.crpix[1]) * self.cdelt[1] + self.crval[1]
        self.z_cart = (self.iz_grid - self.crpix[2]) * self.cdelt[2] + self.crval[2]
        
    def get_pos_polar(self):
        """
        Convert current Cartesian mesh coordinates to Polar coordinates (r, theta, z)
        using the rotation parameters (inc, pa).
        
        Returns:
            tuple: (r, theta, z_polar) arrays
        """
        # x = -pos_cart[XAXIS] * spa + pos_cart[YAXIS] * cpa;
        # y = -(pos_cart[XAXIS] * cpa + pos_cart[YAXIS] * spa) * sinc + pos_cart[ZAXIS] * cinc;
        # z = -(pos_cart[XAXIS] * cpa + pos_cart[YAXIS] * spa) * cinc - pos_cart[ZAXIS] * sinc;
        
        x_rot = -self.x_cart * self.spa + self.y_cart * self.cpa
        y_rot = -(self.x_cart * self.cpa + self.y_cart * self.spa) * self.sinc + self.z_cart * self.cinc
        z_rot = -(self.x_cart * self.cpa + self.y_cart * self.spa) * self.cinc - self.z_cart * self.sinc
        
        r_polar = np.sqrt(x_rot**2 + y_rot**2)
        theta_polar = np.arctan2(y_rot, x_rot)
        
        return r_polar, theta_polar, z_rot

    def set_vel_polar(self, vr, vtheta, vz):
        """
        Convert polar velocities to Cartesian velocities and store in data array.
        
        Args:
            vr: Radial velocity array
            vtheta: Azimuthal velocity array
            vz: Vertical velocity array
        """
        # Re-calculate polar coordinates needed for transformation (or store them if memory allows)
        # For memory efficiency, we might re-calculate or pass theta.
        # But wait, we need theta for the transformation.
        # Assuming we just call get_pos_polar before this or reuse theta.
        
        # Let's re-calculate theta to be safe and self-contained, or optimize later.
        # Actually, let's optimize: get_pos_polar is usually called right before this.
        # But to keep API simple, let's re-compute rotation or allow passing theta.
        # Let's re-compute just the needed parts.
        
        x_rot = -self.x_cart * self.spa + self.y_cart * self.cpa
        y_rot = -(self.x_cart * self.cpa + self.y_cart * self.spa) * self.sinc + self.z_cart * self.cinc
        theta = np.arctan2(y_rot, x_rot)
        
        cth = np.cos(theta)
        sth = np.sin(theta)
        
        # vel_cart[XAXIS] = vel_polar[rAXIS] * (-sth * sinc * cpa - cth * spa) + vel_polar[tAXIS] * (-cth * sinc * cpa + sth * spa) + vel_polar[ZAXIS] * (-cinc * cpa);
        # vel_cart[YAXIS] = vel_polar[rAXIS] * (-sth * sinc * spa + cth * cpa) + vel_polar[tAXIS] * (-cth * sinc * spa - sth * cpa) + vel_polar[ZAXIS] * (-cinc * spa);
        # vel_cart[ZAXIS] = vel_polar[rAXIS] * (sth * cinc) + vel_polar[tAXIS] * (cth * cinc) + vel_polar[ZAXIS] * (-sinc);

        vx = vr * (-sth * self.sinc * self.cpa - cth * self.spa) + \
             vtheta * (-cth * self.sinc * self.cpa + sth * self.spa) + \
             vz * (-self.cinc * self.cpa)
             
        vy = vr * (-sth * self.sinc * self.spa + cth * self.cpa) + \
             vtheta * (-cth * self.sinc * self.spa - sth * self.cpa) + \
             vz * (-self.cinc * self.spa)
             
        vz_cart = vr * (sth * self.cinc) + \
                  vtheta * (cth * self.cinc) + \
                  vz * (-self.sinc)
                  
        self.data[..., VX_VAL] = vx
        self.data[..., VY_VAL] = vy
        self.data[..., VZ_VAL] = vz_cart

    def set_scalar_data(self, density, temperature):
        self.data[..., N_VAL] = density
        self.data[..., T_VAL] = temperature

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import pi, epsilon_0, c
from functions import get_pol_Mie, get_pol_quasistatic, get_red_dielec_fn_Lor

class Emitter:
    def __init__(self, r, p):
        self.r = r
        self.p = p

    def set_Lor_sph_params(self, omega_0, gamma, osc_str, rad):
        self.omega_0 = omega_0
        self.gamma = gamma
        self.osc_str = osc_str
        self.rad = rad

    def pol_Fauche_scalar(self, omega, n_r):
        # scalar
        ci = complex(0, 1)
        gamma = self.gamma
        omega_0 = self.omega_0

        pol = (-3*pi*epsilon_0*gamma*c**3) / omega**3 / (omega - omega_0 - ci*n_r*gamma/2)

        return pol

    def pol_Fauche_matrix(self, omega, n_r):
        # matrix
        p = self.p
        n = p / np.linalg.norm(p)
        pol = self.pol_Fauche_scalar(omega, n_r) * np.outer(n,n)

        return pol

    def get_pol_Lor_sph(self, omega, n_r, eps_inf=1.0):
        # quasistatic dipole polarizability. To include retardation, use get_pol_Mie in function.py
        eps_red = get_red_dielec_fn_Lor(omega, self.omega_0, self.gamma, self.osc_str, n_r, eps_inf)
        s = np.sqrt(eps_red) / n_r
        pol = get_pol_quasistatic(s, self.rad)

        return pol

class Donor(Emitter):
    def __init__(self, r, p):
        Emitter.__init__(self, r, p)

class Acceptor(Emitter):
    def __init__(self, r, p):
        Emitter.__init__(self, r, p)

class Nanoparticle(Emitter):
    def __init__(self, r, R):
        self.r = r # position vector (3,)
        self.R = R # radius

    def set_interp_pol_Mie(self, pol_omega_l, s_l, n_b):
        pol_polr_l = get_pol_Mie(pol_omega_l, s_l, n_b, self.R)
        self.f_interp_pol = interp1d(pol_omega_l, pol_polr_l, kind='linear')

    def set_interp_pol_import(self, pol_omega, pol_polr, pol_kind ='isotropic'):
        if pol_kind == "isotropic":
            self.f_interp_pol = interp1d(pol_omega, pol_polr, kind='linear')
        else:
            raise ValueError("Non-isotropic polarizability not yet supported")

    def get_pos(self):
        return self.r

    def get_rad(self):
        return self.R

    def get_geom_crs(self):
        return pi * self.R**2

    def get_interp_pol_Mie(self, omega):
        return self.f_interp_pol(omega)

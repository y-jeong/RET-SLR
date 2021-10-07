import numpy as np
from scipy.constants import c, epsilon_0, pi

def G_reduced(r_eval, r_source, omega, ref_idx, ref_mu):
    ic = complex(0,1)
    I = np.eye(3)

    r_vec = r_eval - r_source
    R = np.sqrt(np.dot(r_vec, r_vec))
    rhat_rhat = (R**(-2))*np.outer(r_vec, r_vec)
    kb = omega / c * ref_idx
    prefactor = np.exp(ic*kb*R)/(4*pi*R)
    term_1 = (1 + ic/(kb*R) - 1/(kb*R)**2)	
    term_2 = (-1 - 3*ic/(kb*R) + 3/(kb*R)**2)
    G_original = prefactor * (term_1*I + term_2*rhat_rhat)
    G_reduced_ = omega**2 * ref_mu / c**2 / epsilon_0 * G_original

    return G_reduced_

def efield_dipole(r_eval, r_dip, p_dip, omega, nb):
    # curretly doesn't consider the background permeability
    # nb: background refractive index
    ci = complex(0,1)
    kb = omega / c * nb
    r_vec = r_eval - r_dip
    R = np.linalg.norm(r_vec)
    n_vec = r_vec / R

    pre = np.exp(ci*kb*R) / (4*pi*epsilon_0)
    term1 = kb**2 * np.cross(np.cross(n_vec, p_dip), n_vec) / R
    term2 = (3*n_vec*np.dot(n_vec, p_dip) - p_dip) * (1/R**3 - ci*kb/R**2)

    elec = pre * (term1 + term2)
    return elec

def get_red_dielec_fn_Lor(omega, omega_0, gamma, osc_str, n_r, eps_inf=1.0):
    eps_red = eps_inf - osc_str * omega_0**2 / (omega**2 - omega_0**2 + 1j*omega*gamma) # eps/epsilon_0
    return eps_red

def get_pol_quasistatic(s_l, NP_rad):
    # Returns the quasistatic dipole polarizability from the permittivity array input
    pol_polr_l = 4*pi*epsilon_0 * NP_rad**3 * ( (s_l**2-1) / (s_l**2+2) )

    return pol_polr_l

def get_pol_Mie(pol_omega_l, s_l, n_b, a):
    # Returns the fully-retarded dipole polarizability
    k_b_l  = pol_omega_l / c * n_b
    pre_l = 4*pi*epsilon_0 / (2/3*1j) / k_b_l**3
    x_l = k_b_l * a
    sx_l = s_l * x_l 
    numer_l = RB1(x_l) * RB1d(sx_l) - s_l * RB1(sx_l) * RB1d(x_l)
    denom_l = s_l * RB1(sx_l) * RB3d(x_l) - RB3(x_l) * RB1d(sx_l)

    delta_1_l = numer_l / denom_l # The scattering succeptibility of the electric dipole in the Mie theory (Le Ru and Etchegoin)
    pol_polr_l =  pre_l * delta_1_l
    return pol_polr_l

def RB1(x_l):
    # Riccati-Bessel function with the spherical Bessel function of the first kind (n=1)
    result = np.array([ np.sin(x)/x - np.cos(x) for x in x_l ])
    return result

def RB1d(x_l):
    # Differentiated RB1
    result = np.array([ -np.sin(x)/x**2 + np.cos(x)/x + np.sin(x) for x in x_l ])
    return result

def RB3(x_l):
    # Riccati-Bessel function with the spherical Hankel function of the first kind (n=1)
    result = np.array([ complex(np.sin(x)/x - np.cos(x), -np.cos(x)/x - np.sin(x)) for x in x_l ])
    return result

def RB3d(x_l):
    # Differentiated RB3
    result = np.array([ complex(-np.sin(x)/x**2 + np.cos(x)/x + np.sin(x), np.cos(x)/x**2 + np.sin(x)/x - np.cos(x)) for x in x_l ])
    return result

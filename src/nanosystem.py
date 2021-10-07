import numpy as np
from scipy.linalg import solve
from scipy.constants import pi, epsilon_0, c
from functions import G_reduced
from objects import Donor, Acceptor, Nanoparticle

class Nanosystem:
    def __init__(self, don_l, acc_l, NP_l, n_r, mu_r, E0):
        self.don_l = don_l
        self.acc_l = acc_l
        self.NP_l = NP_l
        self.n_r = n_r # background relative refractive index
        self.mu_r = mu_r # background relative permeability
        self.E0 = E0
        self.num_don = np.shape(don_l)[0]
        self.num_acc = np.shape(acc_l)[0]
        self.num_NP = np.shape(NP_l)[0]
        self.num_obj = self.num_NP + self.num_don + self.num_acc

    def get_coupling_factor_1dNP_multiDA_nonpolDA(self, omega, NP_iso_pol=False):
        # output: the list containing a list of the coupling factors for each acceptor
        I = np.eye(3, dtype=float)
        num_NP = self.num_NP
        NP_l = self.NP_l
        don_l = self.don_l
        acc_l = self.acc_l
        num_acc = len(self.acc_l)
        n_r = self.n_r
        mu_r = self.mu_r

        E = np.zeros(3*num_NP, dtype=complex)

        print("coupling factor calculation for wavelength {:.2f} nm".format(2*pi*c/omega*1e9))
        if NP_iso_pol == True:
            for i in np.arange(0, num_NP):
                E[3*i:3*i+3] = np.sum( np.array([ np.dot(G_reduced(NP_l[i].r, don.r, omega, n_r, mu_r), don.p) for don in don_l ]), axis=0 ) # sum of G.p_D over all Donors
            M = self.construct_M1(omega, NP_l, NP_l)
            P = solve(M, E, assume_a='sym') # M is symmetric when the polarizabilities are diagonal

        else:
           raise ValueError("Nonisotropic NP polarizability not yet supported")

        efield_at_acc_total_norm_l = np.array([ self.get_efield_total_norm_nonpolDA(acc.r, P, omega) for acc in acc_l ])
        cpl_fac_l = np.abs( np.array([ np.dot(acc_l[i].p, efield_at_acc_total_norm_l[i]) / np.linalg.norm(acc_l[i].p) for i in np.arange(num_acc) ]) )**2

        return cpl_fac_l
 
    def get_coupling_factor_1dNP_multiDA_polDA(self, omega, NP_iso_pol=False, DA_iso_pol=False):
        # The donors and acceptors are polarizable
        # Depending on if the polarizabilities are isotropic, the interaction matrices and driving terms take slightly different forms
        # output: the list containing a list of the coupling factors for each acceptor
        I = np.eye(3, dtype=float)
        num_NP = self.num_NP
        NP_l = self.NP_l
        don_l = self.don_l
        acc_l = self.acc_l
        num_don = len(self.don_l)
        num_acc = len(self.acc_l)
        n_r = self.n_r
        mu_r = self.mu_r
        num_obj = num_NP + num_don + num_acc

        E = np.zeros(3*num_obj, dtype=complex)
        E_drv = np.zeros(3*num_don, dtype=complex)

        print("coupling factor calculation for wavelength {:.2f} nm".format(2*pi*c/omega*1e9))

        # If all polarizabilities are diagonal, the interaction matrix can take a simpler form (and the driving vector has a correponding form)
        if NP_iso_pol == True and DA_iso_pol == True:
            for i in np.arange(0, num_don):
                E_drv[3*i:3*i+3] = don_l[i].p / don_l[i].get_pol_Lor_sph(omega, n_r, eps_inf=1.0)
            E[3*num_NP:3*(num_NP + num_don)] = E_drv

            M_NP = self.construct_M1(omega, NP_l, NP_l)
            M_D = self.construct_M1(omega, don_l, don_l)
            M_A = self.construct_M1(omega, acc_l, acc_l)
            M_NP_D = self.construct_M1(omega, NP_l, don_l)
            M_NP_A = self.construct_M1(omega, NP_l, acc_l)
            M_D_A = self.construct_M1(omega, don_l, acc_l)

            M = np.vstack( (np.hstack((M_NP, M_NP_D, M_NP_A)), np.hstack((np.transpose(M_NP_D), M_D, M_D_A)), np.hstack((np.transpose(M_NP_A), np.transpose(M_D_A), M_A))) )
            P = solve(M, E, assume_a='sym')

        else:
            for i in np.arange(0, num_don):
                E_drv[3*i:3*i+3] = don_l[i].p
            E[3*num_NP:3*(num_NP + num_don)] = E_drv

            M_NP = self.construct_M2(omega, NP_l, NP_l)
            M_D = self.construct_M2(omega, don_l, don_l)
            M_A = self.construct_M2(omega, acc_l, acc_l)
            M_NP_D = self.construct_M2(omega, NP_l, don_l)
            M_NP_A = self.construct_M2(omega, NP_l, acc_l)
            M_D_A = self.construct_M2(omega, don_l, acc_l)

            M = np.vstack( (np.hstack((M_NP, M_NP_D, M_NP_A)), np.hstack((np.transpose(M_NP_D), M_D, M_D_A)), np.hstack((np.transpose(M_NP_A), np.transpose(M_D_A), M_A))) )
            P = solve(M, E, assume_a='gen')

        efield_at_acc_total_norm_l = np.array([ self.get_efield_total_norm_polDA(acc_idx, P, omega) for acc_idx in np.arange(num_acc) ])

        cpl_fac_l = np.abs( np.array([ np.dot(acc_l[i].p, efield_at_acc_total_norm_l[i]) / np.linalg.norm(acc_l[i].p) for i in np.arange(num_acc) ]) )**2

        return cpl_fac_l
 
    def get_coupling_factor_1dNP_multiDA_Poudel(self, omega):
        # double sum over donors of the terms [n_A G(A,D1) p_D1][n_A G(A,D2) p_D2]
        # and then sum over A's to obtain the total MRET rate
        I = np.eye(3, dtype=float)
        num_NP = self.num_NP
        NP_l = self.NP_l
        don_l = self.don_l
        acc_l = self.acc_l
        num_don = len(self.don_l)
        num_acc = len(self.acc_l)
        n_r = self.n_r
        mu_r = self.mu_r

        E = np.zeros(3*num_NP, dtype=complex)
        M = np.zeros((3*num_NP, 3*num_NP), dtype=complex)

        # Construct M
        # Fill out the diagonal 3 by 3 blocks
        for i in np.arange(0, num_NP):
            pol_NP = self.NP_l[i].f_interp_pol(omega)
            M[3*i:3*i+3, 3*i:3*i+3] = I / pol_NP

        # Fill out the off-diagonal 3 by 3 blocks
        # upper triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*i:3*i+3, 3*j:3*j+3] = (-1) * G_reduced(NP_l[i].r, NP_l[j].r, omega, n_r, mu_r)
        # lower triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*j:3*j+3, 3*i:3*i+3] = M[3*i:3*i+3, 3*j:3*j+3]

        cpl_fac_1st_ll = np.empty(num_don, num_acc) # the order of index: Don, Acc
        for i in np.arange(num_don):
            don = don_l[i]
            # Construct E
            for j in np.arange(0, num_NP):
                E[3*j:3*j+3] = np.dot(G_reduced(NP_l[j].r, don.r, omega, n_r, mu_r), don.p)
            P_singleD = solve(M, E, assume_a='sym') # solve for the induced dipole moments of the NPs for single Donors
            efield_at_acc_total_norm_l = np.array([ self.get_efield_total_norm_singleD(acc.r, P_singleD, don, omega) for acc in acc_l ])
            cpl_fac_1st_ll[i] = np.array([ np.dot(acc_l[j].p, efield_at_acc_total_norm_l[j]) / np.linalg.norm(acc_l[j].p) for j in np.arange(num_acc) ])

        cpl_fac_2nd_l = np.sum( np.array([ np.sum( np.array([ cpl_fac_1st_ll[i,:] * cpl_fac_1st_ll[j,:] for j in num_don ]) ) for i in num_don ]) ) # index: Acc

        return cpl_fac_2nd_l

    def get_coupling_factor_1dNP_singleDA(self, omega):
        I = np.eye(3, dtype=float)
        num_NP = self.num_NP
        NP_l = self.NP_l
        don = self.don_l[0]
        acc = self.acc_l[0]
        n_r = self.n_r
        mu_r = self.mu_r

        E = np.zeros(3*num_NP, dtype=complex)
        M = np.zeros((3*num_NP, 3*num_NP), dtype=complex)

        # Construct E
        for i in np.arange(0, num_NP):
            E[3*i:3*i+3] = np.dot(G_reduced(NP_l[i].r, don.r, omega, n_r, mu_r), don.p)

        # Construct M
        # Fill out the diagonal 3 by 3 blocks
        for i in np.arange(0, num_NP):
            pol_NP = self.NP_l[i].f_interp_pol(omega)
            M[3*i:3*i+3, 3*i:3*i+3] = I / pol_NP

        # Fill out the off-diagonal 3 by 3 blocks
        # upper triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*i:3*i+3, 3*j:3*j+3] = (-1) * G_reduced(NP_l[i].r, NP_l[j].r, omega, n_r, mu_r)
        # lower triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*j:3*j+3, 3*i:3*i+3] = M[3*i:3*i+3, 3*j:3*j+3]

        # Solve
        P = solve(M, E)

        elec_at_acc_total = self.get_efield_total(acc.r, P, omega)
        cpl_fac = np.abs(np.dot(acc.p, elec_at_acc_total))**2 / np.linalg.norm(acc.p)**2 / np.linalg.norm(don.p)**2

        return cpl_fac

    def get_purcell_factor_1dNP_singleD_weak(self, omega):
        I = np.eye(3, dtype=float)
        num_NP = self.num_NP
        NP_l = self.NP_l
        don = self.don_l[0]
        n_don = don.p / np.linalg.norm(don.p)
        n_r = self.n_r
        mu_r = self.mu_r
        kb = omega / c * n_r

        E = np.zeros(3*num_NP, dtype=complex)
        M = np.zeros((3*num_NP, 3*num_NP), dtype=complex)

        # Construct E
        for i in np.arange(0, num_NP):
            E[3*i:3*i+3] = np.dot(G_reduced(NP_l[i].r, don.r, omega, n_r, mu_r), don.p)

        # Construct M
        # Fill out the diagonal 3 by 3 blocks
        for i in np.arange(0, num_NP):
            pol_NP = self.NP_interp_pol_l[i](omega)
            M[3*i:3*i+3, 3*i:3*i+3] = I / pol_NP

        # Fill out the off-diagonal 3 by 3 blocks
        # upper triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*i:3*i+3, 3*j:3*j+3] = (-1) * G_reduced(NP_l[i].r, NP_l[j].r, omega, n_r, mu_r)
        # lower triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*j:3*j+3, 3*i:3*i+3] = M[3*i:3*i+3, 3*j:3*j+3]

        # Solve
        P = solve(M, E) # Need to check if the solution is okay

        efield_scat = self.get_efield_scat(don.r, P, omega) # the scattered field only
        purcell_factor = 1 + 6*pi/(n_r*kb**3) * np.imag(np.dot(n_don, efield_scat))

        return purcell_factor

    def get_purcell_factor_1dNP_singleD_strong_DNP(self, omega):
        # Calculate the Purcell factor, "coupling strongly" the donor and the NPs
        I = np.eye(3, dtype=float)
        num_NP = self.num_NP
        NP_l = self.NP_l
        don = self.don_l[0]
        p_don = don.p
        n_don = p_don / np.linalg.norm(p_don)
        r_don = don.r
        n_r = self.n_r
        mu_r = self.mu_r
        kb = omega / c * n_r

        E = np.zeros(3*num_NP+3, dtype=complex)
        M = np.zeros((3*num_NP+3, 3*num_NP+3), dtype=complex)

        # Construct E
        E[0:3] = p_don # Change it to p_don/beta later
        E[3:] = 0

        # Construct M
        # Fill out the diagonal 3 by 3 blocks
        M[0:3, 0:3] = I / self.don_interp_pol_l[0](omega)
        for i in np.arange(0, num_NP):
            pol_NP = self.NP_interp_pol_l[i](omega)
            
            M[3*i+3:3*i+6, 3*i+3:3*i+6] = I / pol_NP

        # Fill out the off-diagonal 3 by 3 blocks
        # upper triangle
        for i in np.arange(0, num_NP):
            M[0:3, 3*i+3:3*i+6] = (-1) * G_reduced(r_don, NP_l[i].r, omega, n_r, mu_r)
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*i+3:3*i+6, 3*j+3:3*j+6] = (-1) * G_reduced(NP_l[i].r, NP_l[j].r, omega, n_r, mu_r)
        # lower triangle
        for i in np.arange(0, num_NP):
            M[3*i+3:3*i+6, 0:3] = M[0:3, 3*i+3:3*i+6]
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*j+3:3*j+6, 3*i+3:3*i+6] = M[3*i+3:3*i+6, 3*j+3:3*j+6]

        # Solve
        P = solve(M, E)

        efield_scat = self.get_efield_scat(don.r, P[3:], omega) # the scattered field only
        purcell_factor = 1 + 6*pi/(n_r*kb**3) * np.imag(np.dot(n_don, efield_scat)) / np.linalg.norm(p_don)

        return purcell_factor

    # interaction matrix routine for the isotropic polarizability case
    def construct_M1(self, omega, obj_l1, obj_l2):
        I = np.eye(3, dtype=complex)
        n_r = self.n_r
        mu_r = self.mu_r

        len_l1 = len(obj_l1)
        len_l2 = len(obj_l2)

        M_part = np.zeros((3*len_l1, 3*len_l2), dtype=complex)
        M_ij = np.zeros((3,3), dtype=complex)
        if len_l1 != 0 and len_l2 != 0:
            cls_obj_1 = type(obj_l1[0])
            cls_obj_2 = type(obj_l2[0])

            # if a NP-NP, D-D, or A-A interaction matrix
            if cls_obj_1 == cls_obj_2:
                # the diagonal 3 by 3 blocks
                if cls_obj_1 == Nanoparticle:
                    for i in np.arange(0, len_l1):
                        pol = obj_l1[i].f_interp_pol(omega)
                        M_part[3*i:3*i+3, 3*i:3*i+3] = I / pol
                else: # if the objects are donors or acceptors
                    for i in np.arange(0, len_l1):
                        #pol = obj_l1[i].pol_Fauche_scalar(omega, n_r)
                        pol = obj_l1[i].get_pol_Lor_sph(omega, n_r)
                        M_part[3*i:3*i+3, 3*i:3*i+3] = I / pol

                # the off-diagonal 3 by 3 blocks
                for i in np.arange(0, len_l1-1):
                    for j in np.arange(i+1, len_l1):
                        M_part[3*i:3*i+3, 3*j:3*j+3] = (-1) * G_reduced(obj_l1[i].r, obj_l2[j].r, omega, n_r, mu_r)
                        M_part[3*j:3*j+3, 3*i:3*i+3] = M_part[3*i:3*i+3, 3*j:3*j+3]

            # else, if a NP-D, NP-A, or D-A interaction matrix
            else:
                for i in np.arange(0, len_l1):
                    for j in np.arange(0, len_l2):
                        M_part[3*i:3*i+3, 3*j:3*j+3] = (-1) * G_reduced(obj_l1[i].r, obj_l2[j].r, omega, n_r, mu_r)

        return M_part

    def construct_M2(self, omega, obj_l1, obj_l2):
        I = np.eye(3, dtype=complex)
        n_r = self.n_r
        mu_r = self.mu_r

        len_l1 = len(obj_l1)
        len_l2 = len(obj_l2)

        M_part = np.zeros((3*len_l1, 3*len_l2), dtype=complex)
        M_ij = np.zeros((3,3), dtype=complex)

        if len_l1 != 0 and len_l2 != 0:
            cls_obj_1 = type(obj_l1[0])
            cls_obj_2 = type(obj_l2[0])

            pol_l1 = np.empty((len_l1,3,3), dtype=complex)
            if cls_obj_1 == Nanoparticle:
                pol_l1 = np.array([ I * obj_l1[i].get_interp_pol_Mie(omega) for i in np.arange(len_l1) ])
            elif cls_obj_1 == Donor or cls_obj_1 == Acceptor:
                pol_l1 = np.array([ obj_l1[i].get_pol_Lor_sph(omega, n_r) for i in np.arange(len_l1) ])
            else:
                raise ValueError("in construct_M2, unknown cls_obj_1")

            # if a NP-NP, D-D, or A-A interaction matrix
            if cls_obj_1 == cls_obj_2:
                # the diagonal 3 by 3 blocks
                for i in np.arange(len_l1):
                    M_part[3*i:3*i+3, 3*i:3*i+3] = I

                # the off-diagonal 3 by 3 blocks
                for i in np.arange(0, len_l1-1):
                    for j in np.arange(i+1, len_l1):
                        M_part[3*i:3*i+3, 3*j:3*j+3] = (-1) * np.matmul(pol_l1[i], G_reduced(obj_l1[i].r, obj_l2[j].r, omega, n_r, mu_r))
                for i in np.arange(1, len_l1):
                    for j in np.arange(0, i): #
                        M_part[3*i:3*i+3, 3*j:3*j+3] = (-1) * np.matmul(pol_l1[i], G_reduced(obj_l1[i].r, obj_l2[j].r, omega, n_r, mu_r))

            # else, if a NP-D, NP-A, or D-A interaction matrix
            else:
                for i in np.arange(0, len_l1):
                    for j in np.arange(0, len_l2):
                        M_part[3*i:3*i+3, 3*j:3*j+3] = (-1) * np.matmul(pol_l1[i], G_reduced(obj_l1[i].r, obj_l2[j].r, omega, n_r, mu_r))

        return M_part

    def get_ext_crs_tot(self, omega):
        
        I = np.eye(3, dtype=float)
        num_NP = self.num_NP
        NP_l = self.NP_l
        don_l = self.don_l
        acc_l = self.acc_l
        num_acc = len(self.acc_l)
        n_r = self.n_r
        mu_r = self.mu_r
        E0 = self.E0

        E = np.zeros(3*num_NP, dtype=complex)
        M = np.zeros((3*num_NP, 3*num_NP), dtype=complex)

        # Construct E
        for i in np.arange(0, num_NP):
            E[3*i:3*i+3] = E0

        # Construct M
        # Fill out the diagonal 3 by 3 blocks
        for i in np.arange(0, num_NP):
            pol_NP = self.NP_l[i].f_interp_pol(omega)
            M[3*i:3*i+3, 3*i:3*i+3] = I / pol_NP

        # Fill out the off-diagonal 3 by 3 blocks
        # upper triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*i:3*i+3, 3*j:3*j+3] = (-1) * G_reduced(NP_l[i].r, NP_l[j].r, omega, n_r, mu_r)
        # lower triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*j:3*j+3, 3*i:3*i+3] = M[3*i:3*i+3, 3*j:3*j+3]

        # Solve
        P = solve(M, E) # Need to check if the solution is okay

        # Calculate the total extinction cross-section
        pre = omega / (c*epsilon_0) * (np.sqrt(mu_r)/n_r) / np.linalg.norm(E0)**2
        ext_crs_l = np.array([ pre * np.imag( np.dot(P[3*i:3*i+3], E0) ) for i in np.arange(0, num_NP) ])
        ext_crs_tot = np.sum(ext_crs_l)

        return ext_crs_tot
 
    def get_diff_sca_crs_prop(self, num_NP_drv, num_NP_sca, omega, n_vec):
        # Drive a few NPs on the left side. Return the sum of the cross sections of the particle(s) on the right end
        # n_vec is the scattering direction (a unit vector)
        I = np.eye(3, dtype=float)
        num_NP = self.num_NP
        NP_l = self.NP_l
        don_l = self.don_l
        acc_l = self.acc_l
        num_acc = len(self.acc_l)
        n_r = self.n_r
        mu_r = self.mu_r
        E0 = self.E0

        E = np.zeros(3*num_NP, dtype=complex)
        M = np.zeros((3*num_NP, 3*num_NP), dtype=complex)

        # Construct E
        for i in np.arange(0, num_NP_drv):
            E[3*i:3*i+3] = E0

        # Construct M
        # Fill out the diagonal 3 by 3 blocks
        for i in np.arange(0, num_NP):
            pol_NP = self.NP_l[i].f_interp_pol(omega)
            M[3*i:3*i+3, 3*i:3*i+3] = I / pol_NP

        # Fill out the off-diagonal 3 by 3 blocks
        # upper triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*i:3*i+3, 3*j:3*j+3] = (-1) * G_reduced(NP_l[i].r, NP_l[j].r, omega, n_r, mu_r)
        # lower triangle
        for i in np.arange(0, num_NP-1):
            for j in np.arange(i+1, num_NP):
                M[3*j:3*j+3, 3*i:3*i+3] = M[3*i:3*i+3, 3*j:3*j+3]

        # Solve
        P = solve(M, E)

        # Calculate the scattering cross section, expression from PRB 74, 125111 (2006)
        k_r = omega / c * n_r * np.sqrt(mu_r)
        pre = k_r**4 / np.linalg.norm(E0)**2 / (4*pi*epsilon_0)**2 ### Examination needed !!!!!!!!
        diff_sca_crs_l = pre * np.array([ np.linalg.norm( (P[3*i:3*i+3] - n_vec * np.dot(n_vec, P[3*i:3*i+3])) * np.exp(-1j * k_r * np.dot(n_vec, NP_l[i].r)) )**2 for i in np.arange(num_NP-num_NP_sca, num_NP) ]) # only for particles on the right
        diff_sca_crs_prop = np.sum(diff_sca_crs_l)

        return diff_sca_crs_prop

    def get_efield_total(self, r_eval, p_NP_l, omega):
        # Note that p_NP_l is a concatenated array of the induced dipole moments of the NPs. 3N by 1
        NP_l = self.NP_l
        num_NP = len(NP_l)
        don_l = self.don_l
        n_r = self.n_r
        mu_r = self.mu_r

        efield_by_NPs = np.sum( np.array([ np.dot(G_reduced(r_eval, NP_l[i].r, omega, n_r, mu_r), p_NP_l[3*i:3*i+3]) for i in np.arange(num_NP) ]), axis=0 )
        efield_by_Donors = np.sum( np.array([ np.dot(G_reduced(r_eval, don.r, omega, n_r, mu_r), don.p) for don in don_l ]), axis=0 )

        efield = efield_by_NPs + efield_by_Donors

        return efield

    def get_efield_total_norm_nonpolDA(self, r_eval, p_NP_l, omega):
        # P contains all dipole moments (the NPs, the donors, and the acceptors)
        NP_l = self.NP_l
        num_NP = len(NP_l)
        don_l = self.don_l
        n_r = self.n_r
        mu_r = self.mu_r

        efield_by_NPs = np.sum( np.array([ np.dot(G_reduced(r_eval, NP_l[i].r, omega, n_r, mu_r), p_NP_l[3*i:3*i+3]) for i in np.arange(num_NP) ]), axis=0 )
        efield_by_Donors = np.sum( np.array([ np.dot(G_reduced(r_eval, don.r, omega, n_r, mu_r), don.p) for don in don_l ]), axis=0 )

        efield_norm = (efield_by_NPs + efield_by_Donors) / np.linalg.norm(don_l[0].p)

        return efield_norm

    def get_efield_total_norm_polDA(self, acc_idx, P, omega):
        NP_l = self.NP_l
        don_l = self.don_l
        acc_l = self.acc_l
        num_NP = self.num_NP
        num_don = self.num_don
        num_acc = self.num_acc
        n_r = self.n_r
        mu_r = self.mu_r

        r_eval = acc_l[acc_idx].r
        efield_by_NPs = np.sum( np.array([ np.dot(G_reduced(r_eval, NP_l[i].r, omega, n_r, mu_r), P[3*i:3*i+3]) for i in np.arange(num_NP) ]), axis=0 )
        efield_by_Donors = np.sum( np.array([ np.dot(G_reduced(r_eval, don_l[i].r, omega, n_r, mu_r), P[3*(num_NP+i):3*(num_NP+i)+3]) for i in np.arange(num_don) ]), axis=0 )

        efield_by_Acceptors = np.sum( np.array([ np.dot(G_reduced(r_eval, don.r, omega, n_r, mu_r), P[3*(num_NP+num_don+i):3*(num_NP+num_don+i)+3]) for i in np.arange(0, acc_idx) ]), axis=0 )
        efield_by_Acceptors += np.sum( np.array([ np.dot(G_reduced(r_eval, don.r, omega, n_r, mu_r), P[3*(num_NP+num_don+i):3*(num_NP+num_don+i)+3]) for i in np.arange(acc_idx+1, num_acc) ]), axis=0 )

        efield_norm = (efield_by_NPs + efield_by_Donors + efield_by_Acceptors) / np.linalg.norm(don_l[0].p)

        return efield_norm

    def get_efield_total_norm_singleD(self, r_eval, p_NP_l, don, omega):
        NP_l = self.NP_l
        num_NP = len(NP_l)
        don_l = self.don_l
        n_r = self.n_r
        mu_r = self.mu_r

        efield_by_NPs = np.sum( np.array([ np.dot(G_reduced(r_eval, NP_l[i].r, omega, n_r, mu_r), p_NP_l[3*i:3*i+3]) for i in np.arange(num_NP) ]), axis=0 )
        efield_by_Donor = np.dot(G_reduced(r_eval, don.r, omega, n_r, mu_r), don.p)

        efield_norm = (efield_by_NPs + efield_by_Donor) / np.linalg.norm(don.p)

        return efield_norm

    def get_efield_scat(self, r_eval, p_NP_l, omega):
        NP_l = self.NP_l
        don_l = self.don_l
        n_r = self.n_r
        mu_r = self.mu_r

        elec = 0.0
        cnt = 0
        for i in np.arange(len(NP_l)):
            elec += np.dot(G_reduced(r_eval, NP_l[i].r, omega, n_r, mu_r), p_NP_l[3*i:3*i+3])
            cnt += 1

        return elec

    def get_geom_crs_tot(self):
        return self.get_geom_crs_rgt(self.num_NP)

    def get_geom_crs_rgt(self, num_NP_sca):
        return np.sum(np.array([ self.NP_l[i].get_geom_crs() for i in np.arange(self.num_NP - num_NP_sca, self.num_NP) ]))

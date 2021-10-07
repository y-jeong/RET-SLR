# For each given array geometry, this program calculates the extinction cross section, differential scattering cross section (partially driven array), or multi donor-multi acceptor resonance energy transfer (RET) rate

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import pi, c, epsilon_0, e, hbar
from objects import Emitter, Donor, Acceptor, Nanoparticle
from nanosystem import Nanosystem
from functions import get_pol_quasistatic, get_pol_Mie, get_red_dielec_fn_Lor
from constants import dp_au_to_si, nano, micro

if __name__ == "__main__":

    # Define the parameters that define the system to study
    ref_idx_bgr = 1.0 # background refractive index
    prm_bgr = 1.0 # background permeability

    NP_type = "Ag"
    NP_ref_idx_path = "/Users/tigris/Downloads/research/MarcBourgeois/RETSLR/codes/refAg.dat"

    num_NP_l = np.array([81])
    ns_geom_l = np.array([ [ [400*nano,0,40*nano,0,i] ] for i in num_NP_l ])
    ns_geom_l_str = "a400R40".replace(" ", "_")
    # [a,b,R,S,N]: for a sublattice,
    # a: base spacing
    # b: increment of spacing
    # R: base radius
    # S: increment of radius
    # N: number of NPs

    # choose what to calculate and set other system parameters
    calc_opt_l = ["RET_multiDA_polDA", "RET_multiDA_nonpolDA"]
    # ext_crs_tot: all NPs are driven. Total extinction cross section is calculated
    # sca_crs_prop: a specified number of particles on the left are driven. Sum of extinction cross sections on the right side of particles is calculated
    # RET_multiDA: multi-donor, multi-acceptor RET coupling factor
    #    _nonpolDA: non-polarizable donors and acceptors. The dipole moments of the donors are given as the input, and the dipole moments of the NPs are calculated. Redundant inputs such as the donor and acceptor polarizabilities are ignored
    #    _polDA: polarizable donors and acceptors. The driving dipole moments of the donors are given as the input, and then the donor and acceptor polarizabilities are used to calculate the resulting dipole moments of the NPs, donors, and acceptors. Redundant inputs such as the acceptor dipole moments are ignored
    NP_iso_pol = True
    DA_iso_pol = True
    
    E0 = np.array([0.0, 0.0, 1.0]) # external field. Applies to the extinction crs calculation AND scattering (as a result of propagation) crs calculation
    num_NP_drv_l = np.array([40])
    num_NP_sca = 1
    sca_direction = np.array([1.0, 0.0, 0.0]) # This must be a unit vector

    num_don = -1 # number of donors; -1 -> once above each NP
    d_don = 2 * nano # distance from the NP surface; +z direction
    omega_0_don = 2*pi*c/(405e-9) # s-1
    gamma_don = 0.1 * e / hbar # quite arbitrary for now
    osc_str_don = 0.2
    rad_don = 5.0e-9

    num_acc =  1
    d_acc = 2 * nano
    omega_0_acc = 2*pi*c/(405e-9)
    gamma_acc = 0.1 * e / hbar
    osc_str_acc = 0.1
    rad_acc = 10.0e-9

    wvln_begin = 250 * nano # wavelengths are that of in vacuum [nm]
    wvln_end = 650 * nano # Note that the input and the output are in wavelength,
    wvln_cnt = 401       # but the caculation is done in (converted) angular frequency (omega)

    wvln_begin_dielec = 250 * nano
    wvln_end_dielec = 650 * nano
    wvln_cnt_dielec = 401

    # output parameters #
    out_base_dir = "./"
    out_dir = out_base_dir + "RefIdxBgr{:.4f}".format(ref_idx_bgr) + "PrmbtyBgr{:.4f}".format(prm_bgr) + "NPType" + NP_type + "nsGeom" + ns_geom_l_str + "/"
    print(out_dir)
    try:
        os.mkdir(out_dir)
    except OSError:
        print("Creation of the output directory failed")
    else:
        print("Created the working directory {:s}".format(out_dir))

    wvln_l = np.linspace(wvln_begin, wvln_end, wvln_cnt) # for the calculation and plot
    omega_l = 2*pi*c / wvln_l
    wvln_dielec_l = np.linspace(wvln_begin_dielec, wvln_end_dielec, wvln_cnt_dielec) # for the plot of the dielectric function of the donors and acceptors
    omega_dielec_l = 2*pi*c / wvln_dielec_l
    

    ref_idx_data = np.loadtxt(NP_ref_idx_path, skiprows=5)
    ref_idx_wvln_l = ref_idx_data[:,0] * micro # [m]
    ref_idx_omega_l = 2*pi*c / ref_idx_wvln_l
    ref_idx_nk_l = np.array([ complex(ref_idx_data[i,1], ref_idx_data[i,2]) for i in np.arange(np.shape(ref_idx_data)[0]) ])

    s_l = ref_idx_nk_l / ref_idx_bgr
    pol_omega_l = ref_idx_omega_l

    for calc_opt in calc_opt_l:
        if calc_opt == "ext_crs_tot":
            out_fig_path0 = out_dir + "extCrsTot.pdf"
            fig0, ax0 = plt.subplots(1, 1, figsize=(8,6))
            #ax.set_title("", fontsize=24)
            ax0.set_ylabel("Ext. efficiency", fontsize=14)
            ax0.set_xlabel("Wavelength (nm)", fontsize=14)
            ax0.tick_params(axis='x', labelsize=12, labelrotation=0, labelcolor='black')
            ax0.tick_params(axis='y', labelsize=12, labelrotation=0, labelcolor='black')
            #ax.set_yscale('log')

        elif calc_opt == "sca_crs_prop":
            out_fig_path1 = out_dir + "scaCrsProp.pdf"
            fig1, ax1 = plt.subplots(1, 1, figsize=(8,6))
            #ax.set_title("", fontsize=24)
            ax1.set_ylabel("Diff. sca. efficiency", fontsize=14)
            ax1.set_xlabel("Wavelength (nm)", fontsize=14)
            ax1.tick_params(axis='x', labelsize=12, labelrotation=0, labelcolor='black')
            ax1.tick_params(axis='y', labelsize=12, labelrotation=0, labelcolor='black')
            #ax.set_yscale('log')

        elif calc_opt == "RET_multiDA_nonpolDA" or calc_opt == "RET_multiDA_polDA":
            out_base_name2 = "RETMultiDA"
            iso_pol_str2 = "NPIso" + str(NP_iso_pol)
            out_fig_path2 = out_dir + out_base_name2 + iso_pol_str2 + ".pdf"

            fig2, ax2 = plt.subplots(1, 2, figsize=(12,6))
            #ax[0].set_title("", fontsize=24)
            ax2[0].set_ylabel("Coupling factor M ($N^{2} \cdot C^{-4} \cdot m^{-2}$)", fontsize=14)
            ax2[0].set_xlabel("Wavelength (nm)", fontsize=14)
            ax2[0].tick_params(axis='x', labelsize=12, labelrotation=0, labelcolor='black')
            ax2[0].tick_params(axis='y', labelsize=12, labelrotation=0, labelcolor='black')
            ax2[0].set_yscale('log')

            #ax[1].set_title("", fontsize=24)
            ax2[1].set_ylabel("Enhancement factor $M^{NPs}/M^{0}$", fontsize=14)
            ax2[1].set_xlabel("Wavelength (nm)", fontsize=14)
            ax2[1].tick_params(axis='x', labelsize=12, labelrotation=0, labelcolor='black')
            ax2[1].tick_params(axis='y', labelsize=12, labelrotation=0, labelcolor='black')
            ax2[1].set_yscale('log')

        else:
            sys.exit("unknown calc_opt 0")

    ns_cnt = 0
    for ns_geom in ns_geom_l:
        ns_id = "{:d} NPs".format(num_NP_l[ns_cnt])
        print("For {:s}:".format(ns_id))
        NP_l = []
        NP_l0 = [] # an empty NP array for enhancement factor calculation
        num_NP = 0 # Note that this is not necessarily an element of num_NP_l
        cursor = 0 # The position for the beginning of the next sublattice
        for sub_geom in ns_geom:
            a = sub_geom[0]
            b = sub_geom[1]
            R = sub_geom[2]
            S = sub_geom[3]
            N = int(sub_geom[4])
            sub_NP_l = [ Nanoparticle(np.array([cursor + i*a + (i-1)*i/2*b, 0.0, 0.0]), R+i*S) for i in np.arange(N) ]

            NP_l.extend(sub_NP_l)
            cursor += N*a + (N-1)*N/2*b # place the cursor at the position of the "phantom" NP on the right side of the sublattice
            num_NP += N

        # set the dipole polarizabilities of the NPs from the Mie theory
        for NP in NP_l:
            NP.set_interp_pol_Mie(pol_omega_l, s_l, ref_idx_bgr)
        
        for calc_opt in calc_opt_l:
            print("calc_opt: " + calc_opt)
            if calc_opt == "ext_crs_tot":
                # Build the Nanosystem class
                don_l = []
                acc_l = []
                ns = Nanosystem(don_l, acc_l, NP_l, ref_idx_bgr, prm_bgr, E0) # don_l and acc_l are empty

                ns_geom_crs_tot = ns.get_geom_crs_tot()
                ext_crs_tot_l = np.empty(wvln_cnt)
                print("    Calculation of the total extinction cross section:")
                for l in np.arange(wvln_cnt):
                    ext_crs_tot_l[l] = ns.get_ext_crs_tot(omega_l[l])
                ext_eff_tot_l = ext_crs_tot_l / ns_geom_crs_tot
                out_file_ext_crs_path = out_dir + "extCrsTotNS{:04d}.out".format(ns_cnt)
                out_file_ext_crs = open(out_file_ext_crs_path, 'w')
                out_file_ext_crs.write("wavelength (nm)    frequency ($s^-1$)    ext. crs. tot.($m^2$)    ext. eff. tot.\n")
                for l in np.arange(wvln_cnt):
                    out_file_ext_crs.write("{:.4E}         {:.4E}         {:.8E}        {:.8E}\n".format(wvln_l[l], omega_l[l], ext_crs_tot_l[l], ext_eff_tot_l[l]))
                out_file_ext_crs.close()

                line_id = ns_id
                line_ext_eff_tot = ax0.plot(wvln_l/nano, ext_eff_tot_l, '-', label=line_id)

            elif calc_opt == "sca_crs_prop":
                # Build the Nanosystem class
                don_l = []
                acc_l = []
                ns = Nanosystem(don_l, acc_l, NP_l, ref_idx_bgr, prm_bgr, E0) # don_l and acc_l are empty

                ns_geom_crs_prop = ns.get_geom_crs_rgt(num_NP_sca)
                print("    Calculation of the scattering cross section by propagation:")
                for num_NP_drv in num_NP_drv_l:
                    print("        where the left {:d} NPs are driven...".format(num_NP_drv))
                    diff_sca_crs_prop_l = np.empty(wvln_cnt)
                    for l in np.arange(wvln_cnt):
                        diff_sca_crs_prop_l[l] = ns.get_diff_sca_crs_prop(num_NP_drv, num_NP_sca, omega_l[l], sca_direction)
                    diff_sca_eff_prop_l = diff_sca_crs_prop_l / ns_geom_crs_prop

                    out_file_diff_sca_crs_path = out_dir + "DiffScaCrsPropNS{:04d}NumDrvNP{:04d}.out".format(ns_cnt, num_NP_drv)
                    out_file_diff_sca_crs = open(out_file_diff_sca_crs_path, 'w')
                    out_file_diff_sca_crs.write("wavelength (nm)    frequency ($s^-1$)    diff. sca. crs. prop.($m^2$)    diff. sca. eff. prop.\n")
                    for l in np.arange(wvln_cnt):
                        out_file_diff_sca_crs.write("{:.4E}         {:.4E}         {:.8E}        {:.8E}\n".format(wvln_l[l], omega_l[l], diff_sca_crs_prop_l[l], diff_sca_eff_prop_l[l]))
                    out_file_diff_sca_crs.close()

                    line_id = "{:3d} NPs driven".format(num_NP_drv)
                    line_diff_sca_eff_prop = ax1.plot(wvln_l/nano, diff_sca_eff_prop_l, '-', label=line_id)

            elif calc_opt == "RET_multiDA_nonpolDA" or calc_opt == "RET_multiDA_polDA":
                NP_idx_begin = 0
                NP_idx_end = num_NP
                NP_ctr_idx = int( NP_idx_begin + int( (num_NP-1)/2 ) )

                if num_don == -1:
                    don_idx_begin = NP_idx_begin
                    don_idx_end = NP_idx_end
                elif num_don > 0:
                    # The placement of donors are suttle in the case of odd/even or even/odd num_NP and num_don
                    don_idx_begin = int( NP_ctr_idx - int( (num_don-1)/2 ) )
                    don_idx_end = int( NP_ctr_idx + int( num_don/2 ) + 1 )
                else:
                    raise ValueError("invalid num_don")
                don_idx_l = np.arange(don_idx_begin, don_idx_end)

                if num_acc > 0:
                    acc_idx_begin = NP_ctr_idx - int( (num_acc-1)/2 )
                    acc_idx_end = NP_ctr_idx + int( num_acc/2 ) + 1
                else:
                    raise ValueError("invalid num_acc")
                acc_idx_l = np.arange(acc_idx_begin, acc_idx_end)

                r_don_l = np.array([ NP_l[i].get_pos() + np.array([0.0, 0.0, NP_l[i].get_rad()]) + np.array([0.0, 0.0, d_don]) for i in don_idx_l ])
                r_acc_l = np.array([ NP_l[i].get_pos() + np.array([0.0, 0.0,-NP_l[i].get_rad()]) + np.array([0.0, 0.0,-d_acc]) for i in acc_idx_l ])
                p_don_l = np.array([ np.array([ 0.0,  0.0,  5.0]) * dp_au_to_si for i in don_idx_l ]) # donor dipole moments [C m]
                p_acc_l = np.array([ np.array([ 0.0,  0.0, -5.0]) * dp_au_to_si for i in acc_idx_l ]) # acceptor dipole moments [C m]
                omega_0_don_l = np.full(np.shape(don_idx_l)[0], omega_0_don)
                omega_0_acc_l = np.full(np.shape(acc_idx_l)[0], omega_0_acc)
                gamma_don_l = np.full(np.shape(don_idx_l)[0], gamma_don)
                gamma_acc_l = np.full(np.shape(acc_idx_l)[0], gamma_acc)
                osc_str_don_l = np.full(np.shape(don_idx_l)[0], osc_str_don)
                osc_str_acc_l = np.full(np.shape(acc_idx_l)[0], osc_str_acc)
                rad_don_l = np.full(np.shape(don_idx_l)[0], rad_don)
                rad_acc_l = np.full(np.shape(acc_idx_l)[0], rad_acc)

                don_l = [ Donor(r_don_l[i], p_don_l[i]) for i in np.arange(np.shape(don_idx_l)[0]) ]
                for i in np.arange(np.shape(don_idx_l)[0]):
                    don_l[i].set_Lor_sph_params(omega_0_don_l[i], gamma_don_l[i], osc_str_don_l[i], rad_don_l[i])
                acc_l = [ Acceptor(r_acc_l[i], p_acc_l[i]) for i in np.arange(np.shape(acc_idx_l)[0]) ]
                for i in np.arange(np.shape(acc_idx_l)[0]):
                    acc_l[i].set_Lor_sph_params(omega_0_acc_l[i], gamma_acc_l[i], osc_str_acc_l[i], rad_acc_l[i])

                # Build the Nanosystem class
                ns = Nanosystem(don_l, acc_l, NP_l, ref_idx_bgr, prm_bgr, E0)
                ns0 = Nanosystem(don_l, acc_l, NP_l0, ref_idx_bgr, prm_bgr, E0)
                if calc_opt == "RET_multiDA_nonpolDA":
                    line_id_acc = "nonpolarizable DA"
                    # Calculate the list of coupling factors |e_A . E_D|^2 / |p_D|^2 for each acceptor (columns) for each wavelength (rows)
                    cpl_fac_ll = np.array([ ns.get_coupling_factor_1dNP_multiDA_nonpolDA(omega_l[i], NP_iso_pol=NP_iso_pol) for i in np.arange(wvln_cnt) ]) # Note that iso_pol decides the linear algebra routine (since isotropic polarizabilities lead to a symmetric interaction matrix)

                    ## Make the structure with no NPs and calculate coupling factors ##
                    cpl_fac_ll0 = np.array([ ns0.get_coupling_factor_1dNP_multiDA_nonpolDA(omega_l[i], NP_iso_pol=NP_iso_pol) for i in np.arange(wvln_cnt) ])

                else:
                    line_id_acc = "polarizable DA"
                    cpl_fac_ll = np.array([ ns.get_coupling_factor_1dNP_multiDA_polDA(omega_l[i], NP_iso_pol=NP_iso_pol, DA_iso_pol=DA_iso_pol) for i in np.arange(wvln_cnt) ])
                    cpl_fac_ll0 = np.array([ ns0.get_coupling_factor_1dNP_multiDA_polDA(omega_l[i], NP_iso_pol=NP_iso_pol, DA_iso_pol=DA_iso_pol) for i in np.arange(wvln_cnt) ])

                ## Calculation of the enhancement factors ##
                enhan_fac_ll = cpl_fac_ll / cpl_fac_ll0

                # Write the result on the output file
                out_file_path = out_dir + out_base_name2 + iso_pol_str2 + "NS{:04d}.out".format(ns_cnt)

                out_file = open(out_file_path, 'w')
                out_file.write("wavelength (nm)    frequency (s^-1)    coupling factor (N^2 C^-4 m^-2)    enhancement factor\n")
                for i in np.arange(wvln_cnt):
                    out_file.write("{:.4E}         {:.4E}         ".format(wvln_l[i], omega_l[i]) + "  ".join([ "{:.8E}".format(cpl_fac) for cpl_fac in cpl_fac_ll[i] ]) + "        " + "  ".join([ "{:.8E}".format(enhan_fac) for enhan_fac in enhan_fac_ll[i] ]) + "\n")

                out_file.close()

                # Plot
                for acc_idx_zeroed in np.arange(np.shape(acc_idx_l)[0]):
                    line_cpl_fac = ax2[0].plot(wvln_l/nano, cpl_fac_ll[:,acc_idx_zeroed], '-', label=line_id_acc)
                    line_enhan_fac = ax2[1].plot(wvln_l/nano, enhan_fac_ll[:,acc_idx_zeroed], '-', label=line_id_acc)

            else:
                    sys.exit("unknown calc_opt 1")

        ns_cnt += 1

    # Additional plotting options
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.2   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.4   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    #plt.tight_layout()
    #plt.legend(loc="best", fontsize=14)
    for calc_opt in calc_opt_l:
        if calc_opt == "ext_crs_tot":
            line_text_ext_eff = "$n$ = {:.4f}\n$\mu$ = {:.4f}\n".format(ref_idx_bgr, prm_bgr) + NP_type + " NPs" 
            plt.text(0.8, 0.3, line_text_ext_eff, horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=12)
            ax0.legend(loc="best", fontsize=12)
            plt.savefig(out_fig_path0)
            
        elif calc_opt == "sca_crs_prop":
            line_text_diff_sca_eff = "# scattering NPs: {:d}\nscattering direction: {:s}".format(num_NP_sca, " ".join([ str(item) for item in sca_direction ]))
            plt.text(0.8, 0.7, line_text_diff_sca_eff, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)
            ax1.legend(loc="best", fontsize=12)
            plt.savefig(out_fig_path1)

        elif calc_opt == "RET_multiDA_nonpolDA" or calc_opt == "RET_multiDA_polDA":
            ax2[0].legend(loc="best", fontsize=12)
            ax2[1].legend(loc="best", fontsize=12)
            plt.savefig(out_fig_path2)

        else:
            sys.exit("Unknown calc_opt 2") 

    out_base_name4 = "DADielecFn"
    out_fig_path4 = out_dir + out_base_name4 + ".pdf"
    fig4, ax4 = plt.subplots(1, 1, figsize=(8,6))
    ax4t = ax4.twinx()
    ax4.set_ylabel("Re(eps)", fontsize=14, color='red')
    ax4t.set_ylabel("Im(eps)", fontsize=14, color='blue')
    ax4.set_xlabel("Wavelength (nm)", fontsize=14)
    line_text4 = r'$\omega_{0}$' + ": " + "{:.2f}nm".format(2*pi*c/omega_0_don/nano) + "\n" + r'$\gamma$' + ": {:.2f}eV".format(gamma_don/e*hbar) + "\n" + "osc_str: {:.4f}".format(osc_str_don)
    plt.text(0.8, 0.3, line_text4, horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, fontsize=12)

    eps_don = get_red_dielec_fn_Lor(omega_dielec_l, omega_0_don, gamma_don, osc_str_don, ref_idx_bgr, eps_inf=1.0)
    ax4.plot(wvln_dielec_l/nano, np.real(eps_don), color='red')
    ax4t.plot(wvln_dielec_l/nano, np.imag(eps_don), color='blue')
    plt.savefig(out_fig_path4)



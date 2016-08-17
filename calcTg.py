import os
import glob
import argparse
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.get_cmap("viridis") 


global kb
kb = 0.0083145

def get_energies(temp_dirs):
    """Get any energies that exist from subdirectories"""

    frame_idxs = np.concatenate([ np.loadtxt(x + "/inherent_structures/frames_fin.dat", dtype=int) for x in temp_dirs ])
    Etot = np.concatenate([ np.load(x + "/inherent_structures/Etot.npy") for x in temp_dirs ])
    Enat = np.concatenate([ np.load(x + "/inherent_structures/Enat.npy") for x in temp_dirs ])
    Enon = np.concatenate([ np.load(x + "/inherent_structures/Enon.npy") for x in temp_dirs ])

    return Etot, Enat, Enon, frame_idxs

def histogram_energies_in_unfolded_state():
    # Limit
    #Q_U = np.loadtxt("Qtanh_0_05_profile/minima.dat")[0]
    #bounds = (0, Q_U + 5)
    #bounds = (Q_U - 5, Q_U + 5)

    #Q = np.concatenate([ np.load(x + "/Qtanh_0_05.npy") for x in temp_dirs ])
    #U = (Q[frame_idxs] > bounds[0]) & (Q[frame_idxs] < bounds[1])
    #Etot_U = Etot[U]
    #Enat_U = Enat[U]
    #Enon_U = Enon[U]

    # histogram energies. Histogram at fixed native energy.
    nE, bins = np.histogram(Etot - Enon, bins=nbins)
    dE = bins[1] - bins[0]
    probE = nE.astype(float)*dE
    #probE[probE == 0] = np.min(probE[probE != 0])   # THIS LINE IS A PROBLEM
    mid_bin = 0.5*(bins[1:] + bins[:-1])
    minE = np.min(mid_bin)
    mid_bin -= minE

    # remove bins without entries
    nonzero = nE > 100
    mid_bin = mid_bin[nonzero]
    probE = probE[nonzero]

    nE_U, bins_U = np.histogram(Enon, bins=nbins_U, density=True)
    prob_U = np.float(len(Etot_U))/np.float(len(Etot))
    dE_U = bins_U[1] - bins_U[0]
    probE_U = nE_U.astype(float)*dE_U
    #probE_U[probE_U == 0] = np.min(probE_U[probE_U != 0]) # THIS LINE IS A PROBLEM
    mid_bin_U = 0.5*(bins_U[1:] + bins_U[:-1])
    mid_bin_U -= minE

    # remove bins without entries
    nonzero = nE_U > 0
    mid_bin_U = mid_bin_U[nonzero]
    probE_U = probE_U[nonzero]

    # Equation (7) of Nakagawa, Peyrard 2006. Configurational entropy.
    #omegaE = (probE/probE[0])*np.exp(beta*mid_bin)
    #SconfE = np.log(omegaE)
    SconfE = np.log((probE/probE[0])) + beta*mid_bin

    #omegaE_U = ((prob_U*probE_U)/(probE[0]))*np.exp(beta*mid_bin_U)
    #SconfE_U = np.log(omegaE_U)
    SconfE_U = np.log((prob_U*probE_U)/(probE[0])) + beta*mid_bin_U

    # REM fit to the configurational entropy yields Tg!
    coeff = np.polyfit(beta*mid_bin_U, SconfE_U, 2)
    if coeff[0] > 0:
        coeff = np.polyfit(beta*mid_bin_U[5:-5], SconfE_U[5:-5], 2)
    a, b, c = coeff
    E_GS = (-b + np.sqrt(b*b - 4.*a*c))/(2.*a)
    SconfE_U_interp = np.poly1d(coeff)
    dSdE = np.poly1d(np.array([2.*a, b]))
    Tg = 1./(kb*dSdE(E_GS))
    print Tg

    if not os.path.exists("Tg_calc"):
        os.mkdir("Tg_calc")
    os.chdir("Tg_calc")

    # Tg is ~74K using Etot or ~2.5K using Enonnative
    with open("Tg_Enonnative.dat", "w") as fout:
        fout.write("%.2f" % Tg)

    np.savetxt("E_mid_bin.dat", mid_bin)
    np.savetxt("E_mid_bin_U.dat", mid_bin_U)
    np.savetxt("Sconf_tot.dat", SconfE)
    np.savetxt("Sconf_Enon.dat", SconfE_U)

    # solve for other REM parameters using fit.
    Efit = np.linspace(0, beta*max(mid_bin), 1000)
    Tk_line = lambda E: dSdE(E_GS)*E - dSdE(E_GS)*E_GS
    E_Tk_line = np.linspace(0.95*E_GS, 1.01*E_GS, 1000)

    # Save:
    # - Sconf(Enat, Enon)
    # - Enat_grid
    # - Enon_grid

    # Plot
    plt.figure()
    plt.plot(beta*mid_bin, beta*mid_bin, label="$E$")
    plt.plot(beta*mid_bin, SconfE, label="$S_{conf}$")
    plt.plot(beta*mid_bin_U, SconfE_U, label="$S_{conf}(Q_u)$")
    plt.plot(Efit, SconfE_U_interp(Efit), ls='--', lw=1, label="REM fit")
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    
    #plt.annotate('$\\frac{1}{T_g} = \\frac{\\partial S }{\\partial E}$   ', xy=(E_GS, 0),  xycoords='data',
    #        xytext=(0.4, 0.2), textcoords='axes fraction', fontsize=30,
    #        arrowprops=dict(facecolor='gray', shrink=0.01))
    plt.annotate("$T_f = {:.2f} K$\n$T_g = {:.2f} K$".format(T,Tg), xy=(0,0), xycoords="axes fraction",
            xytext=(0.4, 0.8), textcoords='axes fraction', fontsize=22)
    plt.ylabel("Entropy $S_{conf}$ ($k_B$)", fontsize=25)
    plt.xlabel("Energy ($k_B T$)", fontsize=25)
    #plt.title("$T_f = {:.2f}$  $T_g = {:.2f}$".format(T,Tg))
    #plt.title("High frustration   $b = 1.00$", fontsize=25)
    #plt.title("Low frustration   $b = 0.1$", fontsize=25)
    plt.legend(loc=2, fontsize=18)
    plt.xlim(0, beta*np.max(mid_bin))
    plt.ylim(0, beta*np.max(mid_bin))
    plt.savefig("REM_fit_of_dos.png", bbox_inches="tight") 
    plt.savefig("REM_fit_of_dos.pdf", bbox_inches="tight") 
    if display:
        plt.show()

    os.chdir("..")

def REM_Entropy(Enn, Ebar, dE, S_0):
    a = -1./(np.abs(dE)**2)
    b = -Ebar/np.abs(dE)
    c = 0.5*((Ebar/dE)**2) + S_0
    return a*Enn*Enn + b*Enn + c

def thermal_average(T, A, E, S_E, mask_vals):
    return np.sum(A[mask_vals == False]*np.exp(-(E[mask_vals == False] - T*S_E[mask_vals == False])/(kb*T)))

def calculate_thermal_observables():
    
    # Can now use the microcanonical entropy to calculate any quantity over the
    # bins.
    #  - heat capacity vs T
    #  - energy vs T
    
    # DOES NOT WORK AT THE MOMENT
    Tvals = np.linspace(1, 130, 100)

    E_vs_T = np.array([ thermal_average(x, beta*(Etot_grid - (Enat0 + Enon0)),
        Etot_grid - (Enat0 + Enon0), kb*S_Enat_Enon, mask_vals) for x in Tvals ])

    E2_vs_T = np.array([ thermal_average(x, (beta*(Etot_grid - (Enat0 + Enon0)))**2,
        Etot_grid - (Enat0 + Enon0), kb*S_Enat_Enon, mask_vals) for x in Tvals ])

    Cv = (E_vs_T**2 - E2_vs_T)

    varE_vs_T = np.array([ thermal_average(Tvals[x], (beta*(Etot_grid - (Enat0 + Enon0)) - E_vs_T[x])**2,
        Etot_grid - (Enat0 + Enon0), kb*S_Enat_Enon, mask_vals) for x in range(len(Tvals)) ])

    # free energy vs Enat and Enon.
    F_Enat_Enon_therm = beta*(Etot_grid - (Enat0 + Enon0) - T*S_Enat_Enon)
    F_Enat_Enon_therm_ma = np.ma.masked_where(mask_vals, F_Enat_Enon_therm)

    plt.pcolormesh(Enat_grid, Enon_grid, F_Enat_Enon_therm_ma)
    plt.xlabel("$E_{nat}$")
    plt.ylabel("$E_{non}$")
    plt.title("Probability")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inherent structure analysis to get Tg")
    parser.add_argument("--display",
                        action='store_true',
                        help="Don't display plot.")

    parser.add_argument("--n_bins",
                        type=int,
                        default=100,
                        help="Number of bins.")

    args = parser.parse_args()
    display = args.display
    nbins = args.n_bins

    nbins = 100
    nbins_U = 100

    #bins_Enat_Enn = (10, 100) 
    bins_Enat_Enn = (50, 50) 
    nbins_Enat = 20
    nbins_Enon = 20
    
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T = float(fin.read())
    beta = 1./(T*kb)

    temp_dirs = [ "T_{:.2f}_{}".format(T,x) for x in [1,2,3] ]

    Etot, Enat, Enon, frame_idxs = get_energies(temp_dirs)

    # Calculate microcanonical entropy

    P_Enat_Enon, xedges, yedges = np.histogram2d(Enat, Enon, bins=(nbins_Enat, nbins_Enon), normed=True)

    # Get the values of energy for each bin center.
    Enat_mid_bin = 0.5*(xedges[1:] + xedges[:-1])
    Enon_mid_bin = 0.5*(yedges[1:] + yedges[:-1])
    Enat_grid, Enon_grid = np.meshgrid(Enat_mid_bin, Enon_mid_bin)
    Enat_grid = Enat_grid.T
    Enon_grid = Enon_grid.T
    Etot_grid = Enat_grid + Enon_grid

    # Determine the properties of the ground state.
    Enat_0_idx, Enon_0_idx = np.argwhere(P_Enat_Enon == P_Enat_Enon.max())[0]
    Enat0 = Enat_mid_bin[Enat_0_idx]
    Enon0 = Enon_mid_bin[Enon_0_idx]
    P_Enat0_Enn0 = P_Enat_Enon[Enat_0_idx, Enon_0_idx]

    # The formula for density of states can be decomposed into S(Enat, Enon)
    # because Etot = Enat + Enon
    P_Enat_Enon /= P_Enat0_Enn0
    not0 = np.nonzero(P_Enat_Enon)
    S_Enat_Enon = np.nan*np.zeros(P_Enat_Enon.shape)
    S_Enat_Enon[not0] = np.log(P_Enat_Enon[not0]) + beta*(Enat_grid[not0] - Enat0) + beta*(Enon_grid[not0] - Enon0)
    mask_vals = (S_Enat_Enon < 0) | (np.isnan(S_Enat_Enon))
    S_Enat_Enon_ma = np.ma.masked_where(mask_vals, S_Enat_Enon)

    plt.figure()
    plt.pcolormesh(Enat_grid, Enon_grid, S_Enat_Enon_ma, cmap=cmap) # Does this need to be transposed?
    plt.xlabel("$E_{nat}$")
    plt.ylabel("$E_{non}$")
    plt.title("Entropy $S(E_{nat}, E_{non})$")
    plt.colorbar()

    color_idxs = [ float(x)/nbins_Enat for x in range(nbins_Enat) ]

    Tg_Enat = np.zeros(nbins_Enat)
    plt.figure()
    for i in range(nbins_Enat):
        # Fit Random Energy model (parabola) to density of states at each
        # stratum of E_native.
        use_bins = (np.isnan(S_Enat_Enon[i,:]) == False) & (S_Enat_Enon[i,:] > 0)

        if np.sum(use_bins) >= 3:
            S_Enon = S_Enat_Enon[i,use_bins]
            Enon_temp = Enon_mid_bin[use_bins]

            plt.plot(Enon_temp, S_Enon, label="{:.2f}".format(Enat_mid_bin[i]), color=cmap(color_idxs[i]))

            #REM_Entropy(Enn, Ebar, dE, S_0)
            popt, pcov = scipy.optimize.curve_fit(REM_Entropy, Enon_temp, S_Enon, p0=(-10, 10, 100))
            Ebar, dE, S0 = popt
            S_REM = REM_Entropy(Enon_mid_bin, *popt)

            Tg_Enat[i] = dE/np.sqrt(2.*kb*kb*S0)

            # plot the REM fit
            #ymin, ymax = plt.ylim()
            #plt.plot(Enon_mid_bin, S_REM, 'k--')
            #plt.ylim(0, ymax)
    #print Tg_Enat[Tg_Enat > 0]

    #plt.legend(loc=2)
    plt.xlabel("$E_{non}$")
    plt.ylabel("$\\log P(E_{non})$")

    #plt.figure()
    #plt.plot(Enat_mid_bin, Tg_Enat)
    #plt.xlabel("$E_{nat}$")
    #plt.ylabel("$T_g$")
    #plt.title("Glass temperature")
    #plt.show()

    plt.show()

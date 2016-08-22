import os
import glob
import argparse
import numpy as np
import scipy.optimize

global kb
kb = 0.0083145

def get_energies(temp_dirs):
    """Get any energies that exist from subdirectories"""

    frame_idxs = np.concatenate([ np.loadtxt(x + "/inherent_structures/frames_fin.dat", dtype=int) for x in temp_dirs ])
    Etot = np.concatenate([ np.load(x + "/inherent_structures/Etot.npy") for x in temp_dirs ])
    Enat = np.concatenate([ np.load(x + "/inherent_structures/Enat.npy") for x in temp_dirs ])
    Enon = np.concatenate([ np.load(x + "/inherent_structures/Enon.npy") for x in temp_dirs ])

    return Etot, Enat, Enon, frame_idxs

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
#    parser = argparse.ArgumentParser(description="inherent structure analysis to get Tg")
#    parser.add_argument("--display",
#                        action='store_true',
#                        help="Don't display plot.")
#
#    parser.add_argument("--n_bins",
#                        type=int,
#                        default=100,
#                        help="Number of bins.")
#
#    args = parser.parse_args()
#    display = args.display
#    nbins = args.n_bins

    nbins_Enat = 20
    nbins_Enon = 20
    color_idxs = [ float(x)/nbins_Enat for x in range(nbins_Enat) ]
    
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T = float(fin.read())
    beta = 1./(T*kb)

    temp_dirs = [ "T_{:.2f}_{}".format(T,x) for x in [1,2,3] ]

    Etot, Enat, Enon, frame_idxs = get_energies(temp_dirs)

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

    # Calculate microcanonical entropy
    S_Enat_Enon = np.nan*np.zeros(P_Enat_Enon.shape)
    S_Enat_Enon[not0] = np.log(P_Enat_Enon[not0]) + beta*(Enat_grid[not0] - Enat0) + beta*(Enon_grid[not0] - Enon0)
    mask_vals = (S_Enat_Enon < 0) | (np.isnan(S_Enat_Enon))
    S_Enat_Enon_ma = np.ma.masked_where(mask_vals, S_Enat_Enon)

    # calculate Random Energy Model quantities
    Tg_vs_Enat = np.zeros(nbins_Enat)
    Ebar_vs_Enat = np.zeros(nbins_Enat)
    dE_vs_Enat = np.zeros(nbins_Enat)
    S0_vs_Enat = np.zeros(nbins_Enat)
    for i in range(nbins_Enat):
        # Fit Random Energy model (parabola) to microcanonical entropy at each
        # stratum of E_native.
        use_bins = (np.isnan(S_Enat_Enon[i,:]) == False) & (S_Enat_Enon[i,:] > 0)

        if np.sum(use_bins) >= 3:
            S_Enon = S_Enat_Enon[i,use_bins]
            Enon_temp = Enon_mid_bin[use_bins]

            #REM_Entropy(Enn, Ebar, dE, S_0)
            popt, pcov = scipy.optimize.curve_fit(REM_Entropy, Enon_temp, S_Enon, p0=(-10, 10, 100))
            Ebar, dE, S0 = popt
            S_REM = REM_Entropy(Enon_mid_bin, *popt)

            Ebar_vs_Enat[i] = Ebar  
            dE_vs_Enat[i] = dE  
            S0_vs_Enat[i] = S0  
            Tg_vs_Enat[i] = dE/np.sqrt(2.*kb*kb*S0)

#    import matplotlib.pyplot as plt
#    import matplotlib.cm as cm
#    cmap = cm.get_cmap("viridis") 
#
#    # Plot S(E_non) for each stratum of E_nat.
#    plt.figure()
#    for i in range(nbins_Enat):
#        use_bins = (np.isnan(S_Enat_Enon[i,:]) == False) & (S_Enat_Enon[i,:] > 0)
#        if np.sum(use_bins) >= 3:
#            S_Enon = S_Enat_Enon[i,use_bins]
#            Enon_temp = Enon_mid_bin[use_bins]
#            plt.plot(Enon_temp, S_Enon, label="{:.2f}".format(Enat_mid_bin[i]), color=cmap(color_idxs[i]))
#
#    #plt.legend(loc=2)
#    plt.xlabel("$E_{non}$ (k$_B$T)")
#    plt.ylabel("$S(E_{non})$ (k$_B$)")
#
#    #plt.figure()
#    #plt.plot(Enat_mid_bin, Tg_vs_Enat)
#    #plt.xlabel("$E_{nat}$")
#    #plt.ylabel("$T_g$")
#    #plt.title("Glass temperature")
#    #plt.show()
#
#    plt.figure()
#    plt.pcolormesh(Enat_grid, Enon_grid, S_Enat_Enon_ma, cmap=cmap) # Does this need to be transposed?
#    plt.xlabel("$E_{nat}$")
#    plt.ylabel("$E_{non}$")
#    plt.title("Entropy $S(E_{nat}, E_{non})$")
#    plt.colorbar()
#    plt.show()

    if not os.path.exists("Tg_calc"):
        os.mkdir("Tg_calc")
    os.chdir("Tg_calc")
    S_Enat_Enon_ma.dump("S_Enat_Enon.npy")
    np.save("Enat_mid_bin.npy", Enat_mid_bin)
    np.save("Enon_mid_bin.npy", Enon_mid_bin)
    np.save("Ebar_vs_Enat.npy", Ebar_vs_Enat)
    np.save("dE_vs_Enat.npy", dE_vs_Enat)
    np.save("S0_vs_Enat.npy", S0_vs_Enat)
    np.save("Tg_vs_Enat.npy", Tg_vs_Enat)
    os.chdir("..")

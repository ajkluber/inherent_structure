import os
import glob
import argparse
import numpy as np
import scipy.optimize

def thermal_average(T, A, E, S_E, mask_vals):
    return np.sum(A[mask_vals == False]*np.exp(-(E[mask_vals == False] - T*S_E[mask_vals == False])/(kb*T)))

def calculate_Tg_vs_Enat(Enat, Enon, nbins_Enat, nbins_Enon, beta):
    """Calculte Tg as a function of native energy"""
    #P_Enat_Enon, xedges, yedges = np.histogram2d(Enat, Enon, bins=(nbins_Enat, nbins_Enon), normed=True)

    # We mask bins with very low counts to prevent them from influencing the entropy later.
    P_Enat_Enon, xedges, yedges = np.histogram2d(Enat, Enon, bins=(nbins_Enat, nbins_Enon))

    P_Enat_Enon_ma = np.ma.array(P_Enat_Enon, mask=(P_Enat_Enon <= 1))
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    P_Enat_Enon_ma /= np.ma.sum(P_Enat_Enon_ma)   # normalize distribution

    # Get the values of energy for each bin center.
    Enat_mid_bin = 0.5*(xedges[1:] + xedges[:-1])
    Enon_mid_bin = 0.5*(yedges[1:] + yedges[:-1])
    Enat_grid, Enon_grid = np.meshgrid(Enat_mid_bin, Enon_mid_bin)
    Enat_grid = Enat_grid.T
    Enon_grid = Enon_grid.T
    Etot_grid = Enat_grid + Enon_grid

    # Determine the properties of the ground state.
    Enat_0_idx, Enon_0_idx = np.argwhere(P_Enat_Enon_ma == P_Enat_Enon_ma.max())[0]
    Enat0 = Enat_mid_bin[Enat_0_idx]
    Enon0 = Enon_mid_bin[Enon_0_idx]
    P_Enat0_Enn0 = P_Enat_Enon_ma[Enat_0_idx, Enon_0_idx]

    # The formula for density of states can be decomposed into S(Enat, Enon)
    # because Etot = Enat + Enon
    P_Enat_Enon_ma /= P_Enat0_Enn0
    use_vals = (P_Enat_Enon_ma.mask == False)

    # Calculate microcanonical entropy
    S_Enat_Enon = np.zeros(P_Enat_Enon.shape)
    S_Enat_Enon[use_vals] = np.log(P_Enat_Enon[use_vals]) + beta*(Enat_grid[use_vals] - Enat0) + beta*(Enon_grid[use_vals] - Enon0)
    S_Enat_Enon_ma = np.ma.array(S_Enat_Enon, mask=(S_Enat_Enon <= 0))

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

    return Tg_vs_Enat

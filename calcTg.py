import os
import glob
import argparse
import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt

global kb
kb = 0.0083145

def get_frame_idxs(temp_dirs):
    frame_idxs = np.concatenate([ np.loadtxt(x + "/inherent_structures/frames_fin.dat", dtype=int) for x in temp_dirs ])
    return frame_idxs

def get_total_energy(temp_dirs):
    Etot = np.concatenate([ np.load(x + "/inherent_structures/Etot.npy") for x in temp_dirs ])
    return Etot

def get_native_nonnative_energies(temp_dirs):
    """Get any energies that exist from subdirectories"""

    Enat = np.concatenate([ np.load(x + "/inherent_structures/Enat.npy") for x in temp_dirs ])
    Enon = np.concatenate([ np.load(x + "/inherent_structures/Enon.npy") for x in temp_dirs ])

    return Enat, Enon

def REM_Entropy(Enn, Ebar, dE, S_0):
    a = -0.5/(dE**2)
    b = Ebar/(dE**2)
    c = S_0 - 0.5*((Ebar/dE)**2)
    return a*Enn*Enn + b*Enn + c
    #return S_0 - (0.5/(dE**2))*(Enn - Ebar)**2

def parabola(Enn, a, b, c):
    return a*Enn*Enn + b*Enn + c

def thermal_average(T, A, E, S_E, mask_vals):
    return np.sum(A[mask_vals == False]*np.exp(-(E[mask_vals == False] - T*S_E[mask_vals == False])/(kb*T)))

def calculate_Tg_vs_Enat(Enat, Enon, nbins_Enat, nbins_Enon, beta):

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

if __name__ == "__main__":
    nbins_Enat = 70
    nbins_Enon = 20
    
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        Tf = float(fin.read())
    beta = 1./(Tf*kb)

    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]

    Enat, Enon = get_native_nonnative_energies(temp_dirs)

    # calculate Tg as a function of Enat
    #Tg_vs_Enat = calculate_Tg_vs_Enat(Enat, Enon, nbins_Enat, nbins_Enon, beta)
    #print np.abs(Tg_vs_Enat)

    # When there are poor statistics we can estimate Tg using all frames in the
    # unfolded state.
    #Enat = Etot - Enon

    P_Enat, Enat_bins = np.histogram(Enat, bins=nbins_Enat)
    Enat_mid_bin = 0.5*(Enat_bins[1:] + Enat_bins[:-1])

    #plt.figure()
    #plt.plot(Enat_mid_bin, P_Enat)
    #plt.show()

    # max P_Enat above and below median
    peak_idx1 = np.argwhere(P_Enat == np.max(P_Enat[Enat_mid_bin < np.median(Enat)]))[0][0]
    peak_idx2 = np.argwhere(P_Enat == np.max(P_Enat[Enat_mid_bin >= np.median(Enat)]))[0][0]


    # U = an interval around the unfolded state peak in P_Enat
    # Determine the width of the non-native interaction basin -> roughness
    left_side = False
    right_side = False
    threshold = 0.6
    for i in range(len(Enat_mid_bin)):
        if Enat_mid_bin[i] >= np.median(Enat):
            if not left_side:
                if P_Enat[i] >= threshold*P_Enat[peak_idx2]:
                    left_side = i
            if left_side:
                if P_Enat[i] < threshold*P_Enat[peak_idx2]:
                    right_side = i
                    break

    U = (Enat > Enat_mid_bin[left_side]) & (Enat <= Enat_mid_bin[right_side])
    N = (Enat < 0.9*np.min(Enat))
    # Peak to peak distance in P_Enat --> stability gap
    dE_stab = (Enat_mid_bin[peak_idx2] + np.mean(Enon[U])) - (Enat_mid_bin[peak_idx1] + np.mean(Enon[N]))

    P_U = np.sum(U)/float(len(U))
    P_Enon_U, Enon_bins = np.histogram(Enon[U], bins=nbins_Enon)
    Enon_mid_bin = 0.5*(Enon_bins[1:] + Enon_bins[:-1])
    P_Enon_U_ma = np.ma.array(P_Enon_U, mask=(P_Enon_U < 10))
    nonzero = (P_Enon_U > 1)
    #S_Enon_U = np.ma.log(P_Enon_U_ma/P_U) + beta*Enon_mid_bin
    #S_Enon_U = np.ma.log(P_Enon_U[nonzero]/P_U) + beta*Enon_mid_bin[nonzero]


    popt, pcov = scipy.optimize.curve_fit(parabola, Enon_mid_bin[nonzero], np.log(P_Enon_U[nonzero]), p0=(-1, 5, 1))

    #
    a,b,c = popt
    fit_y = parabola(Enon_mid_bin, a, b, c)

    # calculate landscape parameters from fit parameters
    dEnon = np.sqrt(-0.5/a)

    b_prime = b + beta
    c_prime = c + beta*dE_stab

    Ebar = -b_prime/(2.*a)
    S0 = c_prime - b*b/(4*a) + 20

    Tg = dEnon/np.sqrt(2.*kb*kb*S0)

    S_REM = REM_Entropy(Enon_mid_bin, Ebar, dEnon, S0)

    print "Tf = {:5.2f}  Tg = {:5.2f}".format(Tf, Tg)

    if not os.path.exists("Tg_calc"):
        os.mkdir("Tg_calc")
    os.chdir("Tg_calc")
    #S_Enat_Enon_ma.dump("S_Enat_Enon.npy")
    #np.save("Enat_mid_bin.npy", Enat_mid_bin)
    #np.save("Ebar_vs_Enat.npy", Ebar_vs_Enat)
    #np.save("dE_vs_Enat.npy", dE_vs_Enat)
    #np.save("S0_vs_Enat.npy", S0_vs_Enat)
    #np.save("Tg_vs_Enat.npy", Tg_vs_Enat)

    np.save("Enon_mid_bin.npy", Enon_mid_bin)
    np.save("S_REM_Enon.npy", S_REM)

    with open("REM_parameters.dat", "w") as fout:
        fout.write("{:.2f} {:.2f} {:.2f} {:.2f}".format(beta*dE_stab, beta*dEnon, beta*Ebar, S0/kb))
    with open("Tg_Enonnative.dat", "w") as fout:
        fout.write("{:.2f}".format(Tg))
    with open("Tf.dat", "w") as fout:
        fout.write("{:.2f}".format(Tf))
    os.chdir("..")


    #plt.figure()
    #plt.plot(Enon_mid_bin, S_REM, 'k--')
    ##plt.plot(Enon_mid_bin[nonzero], np.log(P_Enon_U[nonzero]), label="data")
    ##plt.plot(Enon_mid_bin, fit_y, 'k--', label="fit")
    ##plt.plot(Enon_mid_bin, S_Enon_U, label="data")
    ##plt.plot(Enon_mid_bin, S_REM, 'k--', label="REM")
    ##plt.title("Tg = {}".format(Tg))
    #plt.xlabel("$E_{non}$")
    #plt.ylabel("$S(E_{non})$")
    #plt.legend()
    #plt.show()

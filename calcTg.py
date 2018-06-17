import os
import glob
import argparse
import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt

import util

global kb
kb = 0.0083145  # KJ/(MOL K)

def REM_Entropy(Enn, Ebar, dE, S_0):
    a = -0.5/(dE**2)
    b = Ebar/(dE**2)
    c = S_0 - 0.5*((Ebar/dE)**2)
    return a*Enn*Enn + b*Enn + c

def calculate_REM(nbins_Enat, nbins_Enon, threshold, save_data=True):
    """Calculate Random Energy Model parameters from inherent structure
    energies"""

    Tf = util.get_T_used()
    beta = 1./(Tf*kb)

    Enat, Enon, Eback = util.get_data(["Enat.npy", "Enon.npy", "Ebackbone.npy"])
    Enat_thm, Enon_thm, Eback_thm = util.get_data(["Enat_thm.npy", "Enon_thm.npy", "Ebackbone_thm.npy"])

    if Enat_thm == []:
        keep = (Enon < 1e1)
    else:
        min_length = min([Enat.shape[0], Enat_thm.shape[0]])
        Enat = Enat[:min_length]
        Enon = Enon[:min_length]
        Eback = Eback[:min_length]
        Enon_thm = Enon_thm[:min_length]
    
        # some energies blow up. Have to be removed.
        keep = (Enon < 1e1) & (Enon_thm < 1e1) 

    Enat = Enat[keep]
    Enon = Enon[keep]
    Eback = Eback[keep]

    Enat_tot = Enat + Eback

    P_Enat, Enat_mid_bin, U, peak_idx1, peak_idx2, left_side, right_side = util.determine_U_frames(Enat_tot, nbins_Enat, threshold=threshold)

    N = (Enat_tot < 0.9*np.min(Enat_tot))
    # Peak to peak distance in P_Enat --> stability gap
    dE_stab = (Enat_mid_bin[peak_idx2] + np.mean(Enon[U])) - (Enat_mid_bin[peak_idx1] + np.mean(Enon[N]))

    # the distribution of non-native energy in the unfolded state determines
    # the unfolded state kinetics.
    P_U = np.sum(U)/float(len(U))
    P_N = np.sum(N)/float(len(U))
    P_Enon_U, Enon_bins = np.histogram(Enon[U], bins=nbins_Enon, density=True)
    Enon_mid_bin = 0.5*(Enon_bins[1:] + Enon_bins[:-1])
    nonzero = P_Enon_U > 10.*P_Enon_U.min()

    # Non-native energy distribution is parabolic according to REM 
    P_ratio = P_Enon_U*P_U/P_N
    x_data = Enon_mid_bin[nonzero]
    y_data = np.log(P_ratio[nonzero])

    # find optimal parameters and uncertainty 
    parabola = lambda Enn, a, b, c: a*Enn*Enn + b*Enn + c
    popt, pcov = scipy.optimize.curve_fit(parabola, x_data, y_data , p0=(-1, 5, 1))
    a, b, c = popt
    a_std, b_std, c_std = np.sqrt(np.diag(pcov))
    y_fit = parabola(Enon_mid_bin, a, b, c)

    # calculate landscape parameters from fit parameters
    dEnon = np.sqrt(-0.5/a)

    b_prime = b + beta
    c_prime = c + beta*dE_stab

    Ebar = -b_prime/(2.*a)

    # there is a shift from the vibrational contribution
    # which is currently unknown.
    S0 = c_prime - b*b/(4*a) + 60

    Tg = dEnon/np.sqrt(2.*kb*kb*S0)

    S_REM = REM_Entropy(Enon_mid_bin, Ebar, dEnon, S0)

    print "Tf = {:5.2f}  Tg = {:5.2f}".format(Tf, Tg)

    #plt.plot(Enon_mid_bin[nonzero], np.log(P_ratio[nonzero]))
    #plt.plot(Enon_mid_bin[nonzero], y_fit[nonzero], 'k--')
    #plt.xlabel(r"$E_{non}$")
    #plt.ylabel(r"$\ln P(E_{non})$")
    #plt.show()

    if save_data:
        if not os.path.exists("Tg_calc"):
            os.mkdir("Tg_calc")
        os.chdir("Tg_calc")
        np.save("Enon_mid_bin.npy", Enon_mid_bin)
        np.save("P_Enon_U.npy", P_Enon_U)
        np.save("S_REM_Enon.npy", S_REM)

        with open("REM_parameters.dat", "w") as fout:
            fout.write("{:.2f} {:.2f} {:.2f} {:.2f}".format(beta*dE_stab, beta*dEnon, beta*Ebar, S0))
        with open("Tg_Enonnative.dat", "w") as fout:
            fout.write("{:.2f}".format(Tg))
        with open("Tf.dat", "w") as fout:
            fout.write("{:.2f}".format(Tf))
        os.chdir("..")

    return (Enat, Enon, Eback), (Enat_thm, Enon_thm, Eback_thm), Enat_mid_bin, P_Enat, left_side, right_side, peak_idx1, peak_idx2, x_data, y_data, Enon_mid_bin, y_fit, nonzero


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate landcape parameters from energy distributions.")
    parser.add_argument("--nbins_Enat", default=70, type=int, help="Number of bins along the native energy")
    parser.add_argument("--nbins_Enon", default=20, type=int, help="Number of bins along the non-native energy")
    parser.add_argument("--threshold", default=0.5, type=float, help="Threshold for U state defintion.")

    args = parser.parse_args()
    nbins_Enat = args.nbins_Enat
    nbins_Enon = args.nbins_Enon
    threshold = args.threshold

    E, E_thm, Enat_mid_bin, P_Enat, left_side, right_side, peak_idx1, peak_idx2, x_data, y_data, Enon_mid_bin, y_fit, nonzero = calculate_REM(nbins_Enat, nbins_Enon, threshold)
    Enat, Enon, Eback = E
    Enat_thm, Enon_thm, Eback_thm = E_thm

#    plt.figure()
#    #plt.plot(Enon_mid_bin, S_REM, 'k--')
#    plt.plot(Enon_mid_bin[nonzero], np.log(P_Enon_U[nonzero]), label="data")
#    plt.plot(Enon_mid_bin, y_fit, 'k--', label="fit")
#    ##plt.plot(Enon_mid_bin, S_Enon_U, label="data")
#    ##plt.plot(Enon_mid_bin, S_REM, 'k--', label="REM")
#    ##plt.title("Tg = {}".format(Tg))
#    #plt.xlabel("$E_{non}$")
#    plt.ylabel("$S(E_{non})$")
#    plt.legend()
#    plt.show()

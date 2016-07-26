import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

global kb
kb = 0.0083145

def get_energies(temps, bounds):
    """Get any energies that exist from subdirectories"""

    Q = np.concatenate([ np.load(x + "/Qtanh_0_05.npy") for x in temps ])
    frames_fin = np.concatenate([ np.loadtxt(x + "/inherent_structures/frames_fin.dat", dtype=int) for x in temps ])
    U = (Q[frames_fin] > bounds[0]) & (Q[frames_fin] < bounds[1])
    Etot = np.concatenate([ np.load(x + "/inherent_structures/Etot.npy") for x in temps ])
    Enat = np.concatenate([ np.load(x + "/inherent_structures/Enat.npy") for x in temps ])
    Enon = np.concatenate([ np.load(x + "/inherent_structures/Enon.npy") for x in temps ])
    Etot_U = Etot[U]
    Enat_U = Enat[U]
    Enon_U = Enon[U]

    return Etot, Enat, Enon, Etot_U, Enat_U, Enon_U, frames_fin

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
    
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T = float(fin.read())
    beta = 1./(T*kb)

    Q_U = np.loadtxt("Qtanh_0_05_profile/minima.dat")[0]
    #bounds = (0, Q_U + 5)
    bounds = (Q_U - 5, Q_U + 5)

    temps = [ "T_{:.2f}_{}".format(T,x) for x in [1,2,3] ]

    Etot, Enat, Enon, Etot_U, Enat_U, Enon_U, frames_fin = get_energies(temps, bounds)

    # histogram energies
    nE, bins = np.histogram(Etot - Enon, bins=nbins)
    dE = bins[1] - bins[0]
    probE = nE.astype(float)*dE
    probE[probE == 0] = np.min(probE[probE != 0])
    mid_bin = 0.5*(bins[1:] + bins[:-1])
    minE = np.min(mid_bin)
    mid_bin -= minE

    nE_U, bins_U = np.histogram(Enon, bins=nbins_U, density=True)
    prob_U = np.float(len(Etot_U))/np.float(len(Etot))
    dE_U = bins_U[1] - bins_U[0]
    probE_U = nE_U.astype(float)*dE_U
    probE_U[probE_U == 0] = np.min(probE_U[probE_U != 0])
    mid_bin_U = 0.5*(bins_U[1:] + bins_U[:-1])
    mid_bin_U -= minE

    # Equation (7) of Nakagawa, Peyrard 2006. Configurational entropy.
    omegaE = (probE/probE[0])*np.exp(beta*mid_bin)
    SconfE = np.log(omegaE)

    omegaE_U = ((prob_U*probE_U)/(probE[0]))*np.exp(beta*mid_bin_U)
    SconfE_U = np.log(omegaE_U)

    # REM fit to the configurational entropy yields Tg!
    coeff = np.polyfit(beta*mid_bin_U, SconfE_U, 2)
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
    np.savetxt("Sconf_tot.dat", SconfE)
    np.savetxt("Sconf_Enon.dat", SconfE_U)

    # solve for other REM parameters using fit.
    Efit = np.linspace(0, beta*max(mid_bin), 1000)
    Tk_line = lambda E: dSdE(E_GS)*E - dSdE(E_GS)*E_GS
    E_Tk_line = np.linspace(0.95*E_GS, 1.01*E_GS, 1000)

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

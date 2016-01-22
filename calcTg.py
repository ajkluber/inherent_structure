import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

global kb
kb = 0.0083145

def get_energy_IS(filename, size, U_bounds, Q, frame_idxs):
    # Load inherent structure energies
    allE = []
    allE_U = []
    for i in range(size):
        if os.path.exists("rank_{}/{}".format(rank, filename)):
            tempE = np.loadtxt("rank_{}/{}".format(rank, filename))
            n_frms = np.min([len(tempE), len(frame_idxs[i])])
            fin_frames = frame_idxs[i][:n_frms]
            U = (Q[fin_frames] > U_bounds[0]) & (Q[fin_frames] < U_bounds[1])
            allE.append(tempE)
            allE_U.append(tempE[U])
        else:
            pass
    if allE == []:
        raise IOError("Some files do not exist.") 
    
    #allE = [ np.loadtxt("rank_{}/{}".format(rank, filename)) for rank in range(size) ]
    #n_finish = [ np.min([len(allE[i]), len(frame_idxs[i])]) for i in range(size) ]
    #frames_fin = [ frame_idxs[i][:n_finish[i]] for i in range(size) ]
    #U = [ ((Q[x] > U_bounds[0]) & (Q[x] < U_bounds[1])) for x in frames_fin ]
    E = np.concatenate(allE)
    E_U = np.concatenate(allE_U)
    return E, E_U

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inherent structure analysis to get Tg")
    parser.add_argument("--size",
                        type=int,
                        required=True,
                        help="Number of subdirs.")

    parser.add_argument("--temperature",
                        type=float,
                        required=True,
                        help="Temperature.")

    parser.add_argument("--n_bins",
                        type=int,
                        default=100,
                        help="Number of bins.")

    parser.add_argument("--plot",
                        action="store_true",
                        help="Plot figure.")

    args = parser.parse_args()
    size = args.size
    T = args.temperature
    nbins = args.n_bins
    plot = args.plot

    #nbins_U = 30
    nbins_U = 100

    beta = 1./(T*kb)

    Q_U = np.loadtxt("../../Qtanh_0_05_profile/minima.dat")[0]
    bounds = (0, Q_U + 5)

    frame_idxs = [ np.loadtxt("rank_{}/frame_idxs.dat".format(rank), dtype=int) for rank in range(size) ]
    Q = np.loadtxt("../Qtanh_0_05.dat")

    all_frame_idxs = np.concatenate(frame_idxs)

    # Thermalized potential energy terms
    #Etot_therm = np.loadtxt("../energyterms.xvg", usecols=(5,))[all_frame_idxs]
    #Eback_therm = np.sum(np.loadtxt("../energyterms.xvg", usecols=(1,2,3)), axis=1)[all_frame_idxs]
    #Enat_therm = np.loadtxt("../Enative.dat")[all_frame_idxs]
    #Enn_therm = np.loadtxt("../Enonnative.dat")[all_frame_idxs]

    # Post-minimization potential energy terms
    Etot, Etot_U = get_energy_IS("Etot.dat", size, bounds, Q, frame_idxs) 
    Enn, Enn_U = get_energy_IS("Enonnative.dat", size, bounds, Q, frame_idxs) 
    #Enat, Enat_U = get_energy_IS("Enative.dat", size, bounds, Q, frame_idxs) 
    #Eback = Etot - Enn - Enat

    # Density of state analysis
    #nE, bins = np.histogram(Etot, bins=nbins)
    nE, bins = np.histogram(Etot - Enn, bins=nbins)
    dE = bins[1] - bins[0]
    probE = nE.astype(float)*dE
    probE[probE == 0] = np.min(probE[probE != 0])
    mid_bin = 0.5*(bins[1:] + bins[:-1])
    minE = np.min(mid_bin)
    mid_bin -= minE

    # Compute Tg using Etot
    #nE_U, bins_U = np.histogram(Etot_U, bins=nbins_U, density=True)
    # OR Compute Tg using Enonnative
    #nE_U, bins_U = np.histogram(Enn_U, bins=nbins_U, density=True)
    nE_U, bins_U = np.histogram(Enn, bins=nbins_U, density=True)
    prob_U = np.float(len(Etot_U))/np.float(len(Etot))
    dE_U = bins_U[1] - bins_U[0]
    probE_U = nE_U.astype(float)*dE_U
    probE_U[probE_U == 0] = np.min(probE_U[probE_U != 0])
    mid_bin_U = 0.5*(bins_U[1:] + bins_U[:-1])
    mid_bin_U -= minE

    # Equation (7) of Nakagawa, Peyrard 2006
    omegaE = (probE/probE[0])*np.exp(beta*mid_bin)
    SconfE = np.log(omegaE)

    #P_E_T = lambda T: np.exp(-(mid_bin - T*0.0083145*SconfE)/(0.0083145*T))

    omegaE_U = ((prob_U*probE_U)/(probE[0]))*np.exp(beta*mid_bin_U)
    SconfE_U = np.log(omegaE_U)

    # REM fit to the configurational entropy yields Tg!
    coeff = np.polyfit(mid_bin_U, SconfE_U, 2)
    a, b, c = coeff
    E_GS = (-b + np.sqrt(b*b - 4.*a*c))/(2.*a)
    SconfE_U_interp = np.poly1d(coeff)
    dSdE = np.poly1d(np.array([2.*a, b]))
    Tg = 1./(kb*dSdE(E_GS))
    print Tg
    # Tg is ~74K using Etot or ~2.5K using Enonnative
    with open("Tg_Enonnative.dat", "w") as fout:
        fout.write("%.2f" % Tg)
    #with open("Tg_Enonnative.dat", "w") as fout:
    #    fout.write("%.2f" % Tg)

    if plot:
        # solve for other REM parameters using fit.
        Efit = np.linspace(0, max(mid_bin), 1000)
        Tk_line = lambda E: dSdE(E_GS)*E - dSdE(E_GS)*E_GS
        E_Tk_line = np.linspace(0.95*E_GS, 1.01*E_GS, 1000)

        # Plot
        plt.figure()
        plt.plot(mid_bin, SconfE, label="$S_{conf}$")
        plt.plot(mid_bin, beta*mid_bin, label="$E$")
        plt.plot(mid_bin_U, SconfE_U, label="$S_{conf}(Q_u)$")
        plt.plot(Efit, SconfE_U_interp(Efit), ls='--', lw=1, label="REM fit")
        #plt.plot(E_Tk_line, Tk_line(E_Tk_line), 'k')
        plt.annotate('$\\frac{1}{T_k} = \\frac{\\partial S }{\\partial E}$', xy=(E_GS, 0),  xycoords='data',
                xytext=(0.4, 0.2), textcoords='axes fraction', fontsize=22,
                arrowprops=dict(facecolor='black', shrink=0.1))
        plt.ylabel("Entropy $S_{conv}$")
        plt.xlabel("Energy")
        plt.title("$T_f = {:.2f}$  $T_k = {:.2f}$".format(T,Tg))
        plt.legend(loc=2)
        plt.ylim(0, np.max(SconfE))
        plt.savefig("REM_fit_of_dos.png", bbox_inches="tight") 
        plt.savefig("REM_fit_of_dos.pdf", bbox_inches="tight") 
        plt.show()




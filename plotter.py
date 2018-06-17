import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import scipy.optimize 

import util
import calcTg 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate landcape parameters from energy distributions.")
    parser.add_argument("--nbins_Enat", default=70, type=int, help="Number of bins along the native energy")
    parser.add_argument("--nbins_Enon", default=20, type=int, help="Number of bins along the non-native energy")
    parser.add_argument("--threshold", default=0.5, type=float, help="Threshold for U state defintion.")
    parser.add_argument("--inset_params", nargs="+", default=[0.22, 0.6, 0.25, 0.25], help="Inset xy and size.")

    args = parser.parse_args()
    nbins_Enat = args.nbins_Enat
    nbins_Enon = args.nbins_Enon
    threshold = args.threshold
    inset_params = np.array(args.inset_params).astype(float)

    E, E_thm, Enat_mid_bin, P_Enat, left_side, right_side, peak_idx1, peak_idx2, x_data, y_data, Enon_mid_bin, y_fit, nonzero  = calcTg.calculate_REM(nbins_Enat, nbins_Enon, threshold, save_data=False)
    Enat, Enon, Eback = E
    Enat_thm, Enon_thm, Eback_thm = E_thm
    Enat_thm_tot = Enat_thm + Eback_thm

    P_Enat_thm, Enat_thm_bins = np.histogram(Enat_thm_tot, bins=nbins_Enat, density=True)
    Enat_thm_mid_bin = 0.5*(Enat_thm_bins[1:] + Enat_thm_bins[:-1])

    fig = plt.figure()
    ax = plt.gca()
    ax.fill_between(Enat_mid_bin, P_Enat, alpha=0.4, color="#5DA5DA")
    ax.fill_between(Enat_thm_mid_bin, P_Enat_thm, alpha=0.4, color="#FAA43A")
    ax.plot(Enat_thm_mid_bin, P_Enat_thm, color="#FAA43A", label=r"$P(E)$")
    ax.plot(Enat_mid_bin, P_Enat, color="#5DA5DA", label=r"$P_{IS}(E)$")
    ax.set_xlabel(r"$E_{nat} + E_{back}$")
    ax.set_ylabel("Probability")
    ax.xaxis.set_ticks_position('bottom')

    ax.legend(loc=1, fontsize=18)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(Enat.min(), xmax)

    # box around the N and U states
    xfill = np.linspace(Enat_mid_bin[left_side], Enat_mid_bin[right_side], 10)
    yfill = ymax*np.ones(len(xfill)) 

    rect_xy = (Enat_mid_bin[left_side - 1], 0)
    width = Enat_mid_bin[right_side] - Enat_mid_bin[left_side - 1]
    height = 1.1*P_Enat[peak_idx2]
    rect_ar = matplotlib.patches.Rectangle(rect_xy, width, height, lw=2, edgecolor='k', fill=False)
    ax.add_artist(rect_ar)
    ax.annotate(r'$U$', xy=(0,0), xytext=(Enat_mid_bin[left_side], 0.8*height),
            xycoords="data", textcoords="data", fontsize=20)

    # inset axes with distribution of non-native energies in the U state
    #left, bottom, width, height = [0.22, 0.6, 0.25, 0.25]
    left, bottom, width, height = inset_params
    ax_inset = fig.add_axes([left, bottom, width, height])
    
    ax_inset.plot(x_data, y_data)
    ax_inset.plot(Enon_mid_bin[nonzero], y_fit[nonzero], 'k--')
    #ax_inset.set_ylabel(r"$\ln\left(\frac{P_{IS}(E_{non})}{P_{IS}(E_{non}^{N})}\right)$")
    ax_inset.set_ylabel(r"$\ln\  P(E_{non}\ |\ U)$", fontsize=16)
    ax_inset.set_xlabel(r"$E_{non}$", fontsize=16)
    #for tick in ax_inset.xaxis.get_major_ticks():
    #    tick.label.set_fontsize(10) 
    #    tick.label.set_rotation("vertical")
    ax_inset.set_yticks([])
    ax_inset.set_xticks([])
    fig.savefig("P_IS_schematic.pdf")
    fig.savefig("P_IS_schematic.png")
    plt.show()

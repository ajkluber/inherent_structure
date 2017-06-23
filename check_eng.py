import numpy as np

import util

def Etot_blowup():
    """Determine if Etot """
    Enat, Enon, Eback = util.get_data(["Enat.npy", "Enon.npy", "Ebackbone.npy"])
    Enat_thm, Enon_thm, Eback_thm = util.get_data(["Enat_thm.npy", "Enon_thm.npy", "Ebackbone_thm.npy"])

    Etot = Enat + Enon + Eback
    Etot_thm = Enat_thm + Enon_thm + Eback_thm
    dE = (Etot.max() - Etot.min())
    dE_thm = (Etot_thm.max() - Etot_thm.min())
    if dE > 1000.:
        blowup = True
    else:
        blowup = False

    return blowup

def is_Etot_minimized():
    """Determine if Etot """
    Enat, Enon, Eback = util.get_data(["Enat.npy", "Enon.npy", "Ebackbone.npy"])
    Enat_thm, Enon_thm, Eback_thm = util.get_data(["Enat_thm.npy", "Enon_thm.npy", "Ebackbone_thm.npy"])

    Etot = Enat + Enon + Eback
    Etot_thm = Enat_thm + Enon_thm + Eback_thm

    return np.all(Etot[1:] < Etot_thm[1:])

def Etot_size():
    """Determine if Etot """
    Enat, Enon, Eback = util.get_data(["Enat.npy", "Enon.npy", "Ebackbone.npy"])
    Enat_thm, Enon_thm, Eback_thm = util.get_data(["Enat_thm.npy", "Enon_thm.npy", "Ebackbone_thm.npy"])

    Etot = Enat + Enon + Eback
    Etot_thm = Enat_thm + Enon_thm + Eback_thm

    return Etot.shape[0] - Etot_thm.shape[0]

if __name__ == "__main__":
    """Check that minimized energies are indeed lower"""
    Enat, Enon, Eback = util.get_data(["Enat.npy", "Enon.npy", "Ebackbone.npy"])
    Enat_thm, Enon_thm, Eback_thm = util.get_data(["Enat_thm.npy", "Enon_thm.npy", "Ebackbone_thm.npy"])

    Etot = Enat + Enon + Eback
    Etot_thm = Enat_thm + Enon_thm + Eback_thm
    E = [Etot, Eback, Enat, Enon]
    E_thm = [Etot_thm, Eback_thm, Enat_thm, Enon_thm]
    Elabels = ["Etot", "Eback", "Enat", "Enon"]
    Elabels_thm = ["Etot_thm", "Eback_thm", "Enat_thm", "Enon_thm"]

    print "{:10}{:>10}{:>10}{:>10}{:>15}".format(" ", "Min", "Max", "shape", "min < thm?")
    for i in range(len(E)):
        Estring = "{:<10}".format(Elabels[i]) 
        Estring += "{:>10.2e}{:>10.2e}{:>10}".format(E[i].min(), E[i].max(), E[i].shape[0]) 

        Estring += "\n{:<10}".format(Elabels_thm[i]) 
        Estring += "{:>10.2e}{:>10.2e}{:>10}".format(E_thm[i].min(), E_thm[i].max(), E_thm[i].shape[0]) 

        #Estring += "\n{:<15}{:>10}".format("min < thm?", str(np.all(Etot[1:] < Etot_thm[1:])))
        Estring += "{:>15}".format(str(np.all(Etot[1:] < Etot_thm[1:])))

        print Estring + "\n"

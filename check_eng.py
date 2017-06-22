import numpy as np

import util

def is_Etot_minimized():
    """Determine if Etot """
    Enat, Enon, Eback = util.get_data(["Enat.npy", "Enon.npy", "Ebackbone.npy"])
    Enat_thm, Enon_thm, Eback_thm = util.get_data(["Enat_thm.npy", "Enon_thm.npy", "Ebackbone_thm.npy"])

    Etot = Enat + Enon + Eback
    Etot_thm = Enat_thm + Enon_thm + Eback_thm

    return np.all(Etot[1:] < Etot_thm[1:])

if __name__ == "__main__":
    """Check that minimized energies are indeed lower"""
    Enat, Enon, Eback = util.get_data(["Enat.npy", "Enon.npy", "Ebackbone.npy"])
    Enat_thm, Enon_thm, Eback_thm = util.get_data(["Enat_thm.npy", "Enon_thm.npy", "Ebackbone_thm.npy"])

    Etot = Enat + Enon + Eback
    Etot_thm = Enat_thm + Enon_thm + Eback_thm

    print "Etot minimized: ", np.all(Etot[1:] < Etot_thm[1:])
    print "Enat minimized: ", np.all(Enat[1:] < Enat_thm[1:])
    print "Enon minimized: ", np.all(Enon[1:] < Enon_thm[1:])
    print "Eback minimized: ", np.all(Eback[1:] < Eback_thm[1:])
    print "Etot (min, max): ", "({}, {})".format(Etot.min(), Etot.max())
    print "Enat (min, max): ", "({}, {})".format(Enat.min(), Enat.max())
    print "Enon (min, max): ", "({}, {})".format(Enon.min(), Enon.max())
    print "Eback (min, max): ", "({}, {})".format(Eback.min(), Eback.max())

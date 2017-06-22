import numpy as np

import util

if __name__ == "__main__":
    """Check that minimized energies are indeed lower"""
    idxs = util.get_data(["frames_fin.dat"])[0]
    Enat, Enon = util.get_data(["Enat.npy", "Enon.npy"])
    Enat_thm, Enon_thm = util.get_data(["Enat_thm.npy", "Enon_thm.npy"])

    print "Enat minimized: ", np.all(Enat[1:] < Enat_thm[1:])
    print "Enon minimized: ", np.all(Enon[1:] < Enon_thm[1:])
    print "Enat (min, max): ", "({}, {})".format(Enat.min(), Enat.max())
    print "Enon (min, max): ", "({}, {})".format(Enon.min(), Enon.max())

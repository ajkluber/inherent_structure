import os
import glob

from simulation.calc.scripts.Enat import calc_Enat_for_directories 
from simulation.calc.scripts.Enn import calc_Enn_for_directories

if __name__ == "__main__":
    import time
    starttime = time.time()

    temps = [ x.rstrip("\n") for x in open("ticatemps", "r").readlines() ]
    path_to_params = "."

    for i in range(len(temps)): 
        os.chdir("{}/inherent_structures".format(temps[i]))
        trajfiles = glob.glob("rank_*/all_frames.xtc")
        calc_Enat_for_directories(trajfiles, path_to_params=path_to_params)
        calc_Enn_for_directories(trajfiles, path_to_params=path_to_params)
        os.chdir("../..")

    print "{} min".format((time.time() - starttime)/60.)

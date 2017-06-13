import os
import glob
import numpy as np
import subprocess as sb

if __name__ == "__main__":
    os.chdir("inherent_structures")
    size = len(glob.glob("rank_*"))

    frame_fin = np.concatenate([ np.loadtxt("rank_" + str(x) + "/frames_fin.dat", dtype=int) for x in range(size) ])
    Etot = np.concatenate([ np.loadtxt("rank_" + str(x) + "/Etot.dat") for x in range(size) ])
    np.savetxt("frames_fin.dat", frame_fin, fmt="%5d")
    np.save("Etot.npy", Etot)

    cat_trajs = " ".join([ "rank_" + str(x) + "/all_frames.xtc" for x in range(size) ])
    with open("trjcat.log", "w") as fout:
        sb.call("trjcat_sbm -f " + cat_trajs + " -o traj.xtc -cat",
            shell=True, stderr=fout, stdout=fout)

    os.chdir("..")

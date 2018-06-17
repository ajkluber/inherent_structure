import os
import time
import glob
import argparse
import numpy as np

import mdtraj as md

import model_builder as mdb

def calculate_energy(model, trajfile, topfile, idxs=[], suffix="", savedir=""):
    """Calculate the energetic terms of a trajectory"""
    if len(idxs) > 0:
        traj = md.load(trajfile, top=topfile)[idxs]
    else:
        traj = md.load(trajfile, top=topfile)

    Ebackbone = model.Hamiltonian.calc_bond_energy(traj) + \
                model.Hamiltonian.calc_angle_energy(traj) + \
                model.Hamiltonian.calc_dihedral_energy(traj) 
    Enat, Enon = model.Hamiltonian.calc_native_nonative_pair_energy(traj, n_native_pairs)
    np.save("{}Ebackbone{}.npy".format(savedir, suffix), Ebackbone)
    np.save("{}Enat{}.npy".format(savedir, suffix), Enat)
    np.save("{}Enon{}.npy".format(savedir, suffix), Enon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy minimization for inherent structure analysis.")
    parser.add_argument("path_to_ini", help="Path to .ini file.")
    parser.add_argument("n_native_pairs", type=int, help="Number of native pairs.")
    parser.add_argument("--trajfile", default="traj.xtc", help="Trajectory filename.")
    parser.add_argument("--topfile", default="ref.pdb", help="Path to topology file.")
    parser.add_argument("--use_ranks", action="store_true", default=True, help="Use trajectories in rank subdirectories.")

    args = parser.parse_args()
    path_to_ini = args.path_to_ini
    n_native_pairs = args.n_native_pairs
    topfile = args.topfile
    trajfile = args.trajfile
    use_ranks = args.use_ranks

    starttime = time.time()

    model, fitopts = mdb.inputs.load_model(path_to_ini)

    with open("Qtanh_0_05_profile/T_used.dat", "r") as fhandle:
        T_used = float(fhandle.read())

    tempdirs = [ "T_{:.2f}_{}".format(T_used, x) for x in [1,2,3] ]

    for i in range(len(tempdirs)):
        os.chdir(tempdirs[i] + "/inherent_structures")
        rank_trajs = glob.glob("rank_*/all_frames.xtc")

        if len(rank_trajs) > 0 and use_ranks:
            for j in range(len(rank_trajs)):
                os.chdir(os.path.dirname(rank_trajs[j]))

                if os.path.exists("frames_fin.dat"):
                    # thermalized energy 
                    idxs = np.loadtxt("frames_fin.dat", dtype=int)
                    calculate_energy(model, "../../" + trajfile, "../" + topfile, idxs=idxs, suffix="_thm", savedir="")

                # minimized energy 
                min_trajfile = os.path.basename(rank_trajs[j])
                calculate_energy(model, min_trajfile, "../" + topfile)

                os.chdir("..")
        elif os.path.exists(trajfile):
            if os.path.exists("frames_fin.dat"):
                # thermalized energy 
                idxs = np.loadtxt("frames_fin.dat", dtype=int)
                calculate_energy(model, "../" + trajfile, topfile, idxs=idxs, suffix="_thm", savedir="")

            # minimized energy 
            calculate_energy(model, trajfile, topfile)
        else:
            raise InputError("Minimized trajectories aren't there")

        os.chdir("../..")

    print "calculation took: {:.4} min".format((time.time() - starttime)/60.)

import os
import time
import argparse
import numpy as np

import mdtraj as md

import model_builder as mdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy minimization for inherent structure analysis.")
    parser.add_argument("path_to_ini", help="Path to .ini file.")
    parser.add_argument("n_native_pairs", type=int, help="Number of native pairs.")
    parser.add_argument("--trajfile", default="traj.xtc", help="Trajectory filename.")
    parser.add_argument("--topfile", default="ref.pdb", help="Path to topology file.")

    args = parser.parse_args()
    path_to_ini = args.path_to_ini
    n_native_pairs = args.n_native_pairs
    topfile = args.topfile
    trajfile = args.trajfile

    starttime = time.time()

    model, fitopts = mdb.inputs.load_model(path_to_ini)

    with open("Qtanh_0_05_profile/T_used.dat", "r") as fhandle:
        T_used = float(fhandle.read())

    tempdirs = [ "T_{:.2f}_{}".format(T_used, x) for x in [1,2,3] ]

    for i in range(len(tempdirs)):
        os.chdir(tempdirs[i] + "/inherent_structures")
        traj = md.load(trajfile, top=topfile)
        Ebackbone = model.Hamiltonian.calc_bond_energy(traj) + \
                    model.Hamiltonian.calc_angle_energy(traj) + \
                    model.Hamiltonian.calc_dihedral_energy(traj) 
        Enat, Enon = model.Hamiltonian.calc_native_nonative_pair_energy(traj, n_native_pairs)
        np.save("Ebackbone.npy", Ebackbone)
        np.save("Enat.npy", Enat)
        np.save("Enon.npy", Enon)
        os.chdir("../..")

    print "calculation took: {:.4} min".format((time.time() - starttime)/60.)

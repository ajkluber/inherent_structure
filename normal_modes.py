import os
import argparse
import numpy as np
import subprocess as sb

import mdtraj 

import simulation.mdp
import simulation.slurm

import model_builder as mdb

def normal_mode_script(path_to_tables, idx):
    script = \
"""#!/bin/bash
grompp_sbm -n ../../index.ndx -f hessian.mdp -c conf.gro -p ../../topol.top -o topol.tpr &> grommp.log
mdrun_sbm -mtx -s topol.tpr -table {0}/table.xvg -tablep {0}/tablep.xvg -tableb {0}/table  &> mdrun.log
g_nmeig_sbm -f nm.mtx -s topol.tpr -ol vals -xvg none  &> g_nmeig.log
#awk '{{ print ($NF) }}' vals.xvg > vals_{1:d}.dat

# cleanup
rm mdout.mdp topol.tpr nm.mtx md.log eigenval.xvg vals.xvg eigenfreq.xvg eigenvec.trr
""".format(path_to_tables, idx)
    return script

  
def run_minimization(frame_idxs, traj, filedir="."):
    """Perform normal mode analysis on each frame"""
    n_frames_out = len(frame_idxs)
    # Minimization needs to be done
    np.savetxt("frame_idxs.dat", frame_idxs, fmt="%d")
    # Loop over trajectory frames
    for i in xrange(traj.n_frames):
        # perform energy minimization using gromacs
        frm = traj.slice(i)
        frm.save_gro("conf.gro")
        script = minimization_script(filedir=filedir)
        with open("minimize.bash", "w") as fout:
            fout.write(script)
        cmd = "bash minimize.bash"
        sb.call(cmd.split())
        # record frame idx
        with open("frames_fin.dat", "a") as fout:
            fout.write("{:d}\n".format(frame_idxs[i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal mode analysis for vibrational free energy.")
    parser.add_argument("--topfile",
                        type=str,
                        required=True,
                        help="Name of topology file. e.g. pdb")

    parser.add_argument("--path_to_tables",
                        type=str,
                        required=True,
                        help="Path to .ini file.")

    parser.add_argument("--trajfile",
                        type=str,
                        default="traj.xtc",
                        help="Name of trajectoy file.")

    parser.add_argument("--serial",
                        action="store_true",
                        help="Use when running serial.")

    args = parser.parse_args()
    topfile = args.topfile
    trajfile = args.trajfile
    path_to_tables = args.path_to_tables
    serial = args.serial

    # Load model and save tables if necessary. NO.

    os.chdir("inherent_structures")

    with open("size", "r") as fin:
        size = int(fin.read())

    if size > 1 and serial:
        print "warning: use same number of processors as generation step for best results."

    if serial:
        # running in serial
        os.chdir("rank_0")

        traj = mdtraj.load(trajfile, top="../../" + topfile)

        if not os.path.exists("nma"):
            os.mkdir("nma")
        os.chdir("nma")

        mdp = simulation.mdp.normal_modes()
        with open("hessian.mdp", "w") as fout:
            fout.write(mdp)

        for i in range(traj.n_frames):
            print i
            frm = traj.slice(i)
            frm.save_gro("conf.gro")
            with open("normal.bash", "w") as fout:
                fout.write(normal_mode_script(path_to_tables, i))
            sb.call("bash normal.bash", shell=True)

            #with open("frames_fin.dat", "a") as fout:
            #    fout.write("{:d}\n".format(frame_idxs[i]))
             
            # TODO: concatenate eigenvals in one file.
            with open("vals.xvg", "r") as fin:
                eigenvals = [ " ".join(fin.readlines()) ]
            

        os.chdir("..")
        os.chdir("..")
    else:
        # running in parallel
        from mpi4py import MPI
        comm = MPI.COMM_WORLD   
        size = comm.Get_size()  
        rank = comm.Get_rank()
        pass

    os.chdir("..")

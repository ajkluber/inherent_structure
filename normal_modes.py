import os
import glob
import time
import argparse
import numpy as np
import subprocess as sb

import mdtraj as md

import simulation.gromacs.mdp
import simulation.slurm

import model_builder as mdb

def normal_mode_script(path_to_tables):
    script = \
"""#!/bin/bash
grompp_sbm -n ../../index.ndx -f ../hessian.mdp -c conf.gro -p ../../topol.top -o topol.tpr &> grommp.log
mdrun_sbm -mtx -s topol.tpr -table {0}/table.xvg -tablep {0}/tablep.xvg -tableb {0}/table  &> mdrun.log
g_nmeig_sbm -f nm.mtx -s topol.tpr -ol vals -xvg none  &> g_nmeig.log
awk '{{ print ($NF) }}' vals.xvg > vals.dat

# cleanup
rm mdout.mdp topol.tpr nm.mtx md.log eigenval.xvg vals.xvg eigenfreq.xvg eigenvec.trr
""".format(path_to_tables)
    return script

def prep_normal_mode(path_to_ini, path_to_py=""):
    """Prepare for normal mode analysis. 
    
    Parameters
    ----------
    path_to_ini : str
        File path to the '.ini' file for loading the simulation model
    path_to_py : str, opt.
        File path to a script augment the simulation model (e.x. add non
        standard interactions). 
    """

    ini_dir, ini_file = os.path.split(path_to_ini)
    path_to_tables = ini_dir + "/tables"

    if not os.path.exists("inherent_structures"):
        raise IOError("No inherent_structures directory found!")
    if not os.path.exists(path_to_tables):
        os.mkdir(path_to_tables)

    # save simulation files and normal mode analysis protocol.
    mdp = simulation.gromacs.mdp.normal_modes()
    with open("hessian.mdp", "w") as fout:
        fout.write(mdp)

    # load model
    cwd = os.getcwd()
    os.chdir(ini_dir)
    model, fitopts = mdb.inputs.load_model(ini_file)

    if path_to_py != "":
        import imp
        modify_py = imp.load_source("DUMMY", path_to_py)
        model = modify_py.augment_model(model)
    os.chdir(cwd)

    # save model files
    writer = mdb.models.output.GromacsFiles(model)
    writer.write_simulation_files(path_to_tables=path_to_tables)
    os.chdir("..")

def run_nma(path_to_tables, frame_idxs, traj, max_time, start_time):
    """Perform energy minimization on each frame
    
    Parameters
    ----------
    path_to_tables : str
        Full path to the directory that holds the tables/ subdirectory.
    frame_idxs : np.ndarray (n_frames,)
        Indices of frames in input trajectory minimized.
    traj : obj. mdtraj.Trajectory
        The minimized trajectory.
    max_time : float
        The maximum runtime. Will terminate if runtime exceeds this max.
    start_time : float
        The starting time in seconds. Used to calculate runtime.
    """

    if os.path.exists("hess_frames_fin.dat"):
        # restart calculation
        start_idx = len(np.loadtxt("hess_frames_fin.dat", dtype=int))
    else:
        # start from the beginning of traj
        start_idx = 0

    for i in xrange(start_idx, traj.n_frames):
        dtime = time.time() - start_time
        if dtime > max_time:
            print "time limit of {:.2f} min exceeded, exiting".format(max_time/60.)
            raise SystemExit
        else:
            # diagonalize Hessian using gromacs
            frm = traj.slice(i)
            frm.save_gro("conf.gro")
            script = normal_mode_script(path_to_tables)
            with open("normal.bash", "w") as fout:
                fout.write(script)
            cmd = "bash normal.bash"
            sb.call(cmd.split())

            with open("vals.dat", "r") as fin:
                eigenvals = " ".join([ x.rstrip("\n") for x in fin.readlines()]) 

            with open("all_vals.dat", "a") as fout:
                fout.write(eigenvals + "\n")

            # record finished frame idxs
            with open("hess_frames_fin.dat", "a") as fout:
                fout.write("{:d}\n".format(frame_idxs[i]))
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal mode analysis for vibrational free energy.")
    parser.add_argument("--path_to_tables",
                        type=str,
                        required=True,
                        help="Path to .ini file.")

    parser.add_argument("--topfile",
                        type=str,
                        default="ref.pdb",
                        help="Name of topology file. e.g. pdb")

    parser.add_argument("--trajfile",
                        type=str,
                        default="traj.xtc",
                        help="Name of trajectoy file.")
    
    parser.add_argument("--path_to_py",
                        type=str,
                        default="",
                        help="Path to .py script that modifies model.")

    parser.add_argument("--serial",
                        action="store_true",
                        help="Use when running serial.")

    parser.add_argument("--max_time",
                        default=0.995*8*60*60.,
                        type=float,
                        help="Maximum runtime in secs.")

    # Performance on one processor is roughly 60sec/frame. So 1proc can do
    # about 480 frames over 8hours. Should use the same number of processors
    # as the generate step.

    starttime = time.time()

    args = parser.parse_args()
    topfile = args.topfile
    trajfile = args.trajfile
    path_to_tables = args.path_to_tables
    serial = args.serial
    max_time = args.max_time
    path_to_py = args.path_to_py

    if path_to_py != "":
        path_to_py = os.path.abspath(args.path_to_py)
        if not os.path.exists(path_to_py):
            raise IOError(path_to_py + " does not exist!")

    if serial:
        #for i in range(len(frames_fin)):
        prep_normal_mode(path_to_ini, stride, size=size, path_to_py=path_to_py)
        os.chdir("inherent_structures")

        with open("size", "r") as fin:
            size = int(fin.read())

        if size > 1:
            raise IOError("Need to use same number of processors as generation step.") 

        os.chdir("rank_0")
        traj = md.load(trajfile, top="../../" + topfile)
        frame_idxs = np.loadtxt("frames_fin.dat", dtype=int)

        run_nma(path_to_tables, frame_idxs, traj, max_time, start_time)
        os.chdir("../..")
    else:
        # running in parallel
        from mpi4py import MPI
        comm = MPI.COMM_WORLD   
        size = comm.Get_size()  
        rank = comm.Get_rank()

        if rank == 0:
            prep_normal_mode(path_to_ini, stride, size=size, path_to_py=path_to_py)

        comm.Barrier()

        if not os.path.exists("rank_{}".format(rank)):
            os.mkdir("rank_{}".format(rank))
        os.chdir("rank_{}".format(rank))
        traj = md.load(trajfile, top="../../" + topfile)
        frame_idxs = np.loadtxt("frames_fin.dat", dtype=int)

        run_nma(path_to_tables, frame_idxs, traj, max_time, start_time)
        os.chdir("../..")

    #stoptime = time.time()
    #print "took: {:.2f} min".format(((stoptime - starttime)/60.))

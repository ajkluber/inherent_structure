import os
import sys
import time
import argparse
import numpy as np
import subprocess as sb

import mdtraj as md

import simulation.gromacs.mdp
import model_builder as mdb

def minimization_script(path_to_tables):
    """Takes a starting structure and energy minimizes it."""
    script = \
"""#!/bin/bash
# run energy minimization
grompp_sbm -n ../index.ndx -f ../em.mdp -c conf.gro -p ../topol.top -o topol_4.5.tpr &> grompp.log
mdrun_sbm -nt 1 -s topol_4.5.tpr -table {0}/table.xvg -tablep {0}/tablep.xvg -tableb {0}/table &> mdrun.log

# get final energy 
g_energy_sbm -f ener.edr -o Energy -xvg none &> energy.log << EOF
Potential
EOF
tail -n 1 Energy.xvg | awk '{{ print $(NF) }}' >> Etot.dat

# get final structure 
fstep=`grep "Low-Memory BFGS Minimizer converged" md.log | awk '{{ print $(NF-1) }}'`
trjconv_sbm -f traj.trr -s topol_4.5.tpr -n ../index.ndx -o temp_frame.xtc -dump ${{fstep}} &> trjconv.log << EOF
System
EOF

# concatenate to trajectory
if [ ! -e all_frames.xtc ]; then
    mv temp_frame.xtc all_frames.xtc
else
    trjcat_sbm -f all_frames.xtc temp_frame.xtc -o all_frames.xtc -n ../index.ndx -nosort -cat &> trjcat.log << EOF
System
EOF
    rm temp_frame.xtc
fi

# cleanup
rm conf.gro mdout.mdp topol_4.5.tpr traj.trr md.log ener.edr confout.gro Energy.xvg
""".format(path_to_tables)
    return script

def prep_minimization(path_to_ini, stride, size=1, path_to_py=""):
    """Prepare for minimization
    
    Parameters
    ----------
    path_to_ini : str
        File path to the '.ini' file for loading the simulation model
    stride : int
        Number of frames to skip.
    size : int, opt. 
        Number of processors used in calculation
    path_to_py : str, opt.
        File path to a script augment the simulation model (e.x. add non
        standard interactions). 
    """
    ini_dir, ini_file = os.path.split(path_to_ini)
    path_to_tables = ini_dir + "/tables"

    if not os.path.exists("inherent_structures"):
        os.mkdir("inherent_structures")
    if not os.path.exists(path_to_tables):
        os.mkdir(path_to_tables)

    with open("stride", "w") as fout:
        fout.write(str(stride))

    with open("size", "w") as fout:
        fout.write(str(size))
    
    # save simulation files and energy minimization protocol.
    os.chdir("inherent_structures")
    mdp = simulation.gromacs.mdp.energy_minimization()
    with open("em.mdp", "w") as fout:
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

def run_minimization(path_to_tables, frame_idxs, traj, max_time, start_time):
    """Perform energy minimization on each frame
    
    Parameters
    ----------
    path_to_tables : str
        Full path to the directory that holds the tables/ subdirectory.
    frame_idxs : np.ndarray (n_frames,)
        Indices of frames that will be minimized.
    traj : obj. mdtraj.Trajectory
        The thermalized trajectory to be minimize.
    max_time : float
        The maximum runtime. Will terminate if runtime exceeds this max.
    start_time : float
        The starting time in seconds. Used to calculate runtime.
    """

    if os.path.exists("frame_idxs.dat") and os.path.exists("frames_fin.dat"):
        # restart calculation
        temp_frame_idxs = np.loadtxt("frame_idxs.dat", dtype=int)
        assert np.allclose(frame_idxs, temp_frame_idxs), "Need to use the same stride and nproc to restart."
        frames_fin = np.loadtxt("frames_fin.dat", dtype=int)
        start_idx = len(frames_fin)
    else:
        # minimize from the beginning of the chunk
        start_idx = 0
        np.savetxt("frame_idxs.dat", frame_idxs, fmt="%d")

    for i in xrange(start_idx, traj.n_frames):
        dtime = time.time() - start_time
        if dtime > max_time:
            print "time limit of {:.2f} min exceeded, exiting".format(max_time/60.)
            raise SystemExit
        else:
            # perform energy minimization using gromacs
            frm = traj.slice(i)
            frm.save_gro("conf.gro")
            script = minimization_script(path_to_tables)
            with open("minimize.bash", "w") as fout:
                fout.write(script)
            cmd = "bash minimize.bash"
            sb.call(cmd.split())

            # record frame idx
            with open("frames_fin.dat", "a") as fout:
                fout.write(str(frame_idxs[i]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy minimization for inherent structure analysis.")
    parser.add_argument("--topfile",
                        type=str,
                        required=True,
                        help="Name of topology file. e.g. pdb")

    parser.add_argument("--path_to_ini",
                        type=str,
                        required=True,
                        help="Path to .ini file.")

    parser.add_argument("--n_frames",
                        type=int,
                        required=True,
                        help="Number of frames in trajectory.")
    
    parser.add_argument("--trajfile",
                        type=str,
                        default="traj.xtc",
                        help="Name of trajectoy file.")

    parser.add_argument("--stride",
                        type=int,
                        default=10,
                        help="Number of frames to stride. Subsample.")

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

    # Performance on one processor is roughly 11-40sec/frame. So 1proc can do
    # about 2500 frames over 8hours. Adjust the number of processors (size)
    # and subsample (stride) accordingly

    start_time = time.time()
    
    args = parser.parse_args()
    topfile = args.topfile
    trajfile = args.trajfile
    path_to_ini = args.path_to_ini
    n_frames = args.n_frames
    stride = args.stride
    serial = args.serial
    max_time = args.max_time
    path_to_py = args.path_to_py

    if path_to_py != "":
        path_to_py = os.path.abspath(args.path_to_py)
        if not os.path.exists(path_to_py):
            raise IOError(path_to_py + " does not exist!")

    path_to_tables = os.path.dirname(path_to_ini) + "/tables"

    if serial:
        # when running on only one processor
        prep_minimization(path_to_ini, stride, path_to_py=path_to_py)
        os.chdir("inherent_structures")
        with open("size", "w") as fout:
            fout.write(str(1))

        # run minimization
        frame_idxs = np.arange(0, n_frames, stride)
        traj = md.load("../" + trajfile, top="../" + topfile, stride=stride)
        if not os.path.exists("rank_0"):
            os.mkdir("rank_0")
        os.chdir("rank_0")
        run_minimization(path_to_tables, frame_idxs, traj, max_time, start_time)
        #sb.call("mv all_frames.xtc ../traj.xtc", shell=True)
        #sb.call("cp frames_fin.dat ../frames_fin.dat", shell=True)
        os.chdir("../..")
    else:
        # running in parallel
        from mpi4py import MPI
        comm = MPI.COMM_WORLD   
        size = comm.Get_size()  
        rank = comm.Get_rank()

        if rank == 0:
            prep_minimization(path_to_ini, stride, size=size, path_to_py=path_to_py)

        comm.Barrier()

        os.chdir("inherent_structures")
        if rank == 0:
            # distribute trajectory chunks to all processors. This reduces
            # memory usage, because only one chunk is loaded into memory at a
            # time.
            with open("size", "w") as fout:
                fout.write(str(size))

            chunksize = n_frames/size
            if (n_frames % size) != 0:
                chunksize += 1

            rank_i = 0
            start_idx = 0
            for chunk in md.iterload("../" + trajfile, top="../" + topfile, chunk=chunksize):
                chunk_idxs = np.arange(0, chunk.n_frames, stride)
                sub_chunk = chunk.slice(chunk_idxs)

                if (rank_i == 0) and (rank == 0):
                    frame_idxs = chunk_idxs + start_idx
                    traj = sub_chunk
                else:
                    comm.send(chunk_idxs + start_idx, dest=rank_i, tag=7)
                    comm.send(sub_chunk, dest=rank_i, tag=11)
                rank_i += 1
                start_idx += chunk.n_frames
            
        if rank > 0:
            # receive trajectory chunk and corresponding frame indices.
            frame_idxs = comm.recv(source=0, tag=7)
            traj = comm.recv(source=0, tag=11)

        if not os.path.exists("rank_{}".format(rank)):
            os.mkdir("rank_{}".format(rank))
        os.chdir("rank_{}".format(rank))
        run_minimization(path_to_tables, frame_idxs, traj, max_time, start_time)
        os.chdir("..")

#        # * Easier to leave them as separate trajectories for restarting
#        # * calculation and further analysis.
#
#        # If all trajectories finished then bring them together.
#        comm.Barrier()
#
#        if rank == 0:
#            frames_fin = np.concatenate([ np.loadtxt("rank_" + str(x) + "/frames_fin.dat", dtype=int) for x in range(size) ])
#            Etot = np.concatenate([ np.loadtxt("rank_" + str(x) + "/Etot.dat") for x in range(size) ])
#            np.savetxt("frames_fin.dat", frames_fin, fmt="%5d")
#            np.save("Etot.npy", Etot)
#
#            cat_trajs = " ".join([ "rank_" + str(x) + "/all_frames.xtc" for x in range(size) ])
#            with open("trjcat.log", "w") as fout:
#                sb.call("trjcat_sbm -f " + cat_trajs + " -o traj.xtc -cat",
#                    shell=True, stderr=fout, stdout=fout)
        os.chdir("..")

import os
import argparse
import numpy as np
import subprocess as sb

import mdtraj as md

import simulation.mdp
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
    mdp = simulation.mdp.energy_minimization()
    with open("em.mdp", "w") as fout:
        fout.write(mdp)

    # load model
    cwd = os.getcwd()
    os.chdir(ini_dir)
    model, fitopts = mdb.inputs.load_model(ini_file)
    if path_to_py != "":
        if not os.path.exists(path_to_py):
            raise IOError(path_to_py + " does not exist!")
        else:
            os.chdir(os.path.dirname(path_to_py))
            import importlib
            modify_py = importlib.import_module(os.path.basename(path_to_py).split(".py")[0])
            model = modify_py.augment_model(model)
    os.chdir(cwd)

    # save model files
    writer = mdb.models.output.GromacsFiles(model)
    writer.write_simulation_files(path_to_tables=path_to_tables)
    os.chdir("..")

def run_minimization(path_to_tables, frame_idxs, traj):
    """Perform energy minimization on each frame
    
    Parameters
    ----------
    path_to_tables : str
        Full path to the directory that holds the tables/ subdirectory.
    frame_idxs : np.ndarray (n_frames,)
        Indices of frames that will be minimized.
    traj : obj. mdtraj.Trajectory
        The thermalized trajectory to be minimize.
    """

    # Minimization needs to be done
    np.savetxt("frame_idxs.dat", frame_idxs, fmt="%d")
    # Loop over trajectory frames
    for i in xrange(traj.n_frames):
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

def restart(trajfile, topology):
    """TODO: write restart function"""
    # In progress
    frame_idxs = np.loadtxt("frame_idxs.dat", dtype=int)
    frame_fin = np.loadtxt("frames_fin.dat", dtype=int)

    if len(frame_fin) < len(frame_idxs):
        # minimize frames that were not finished
        traj_full = md.load(trajfile, top=topology)
        traj = traj_full.slice(frame_idxs)

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

    parser.add_argument("--mpi",
                        action="store_true",
                        help="Use when running in parallel.")

    # Performance on one processor is roughly 11-40sec/frame. 
    # So 1proc can do about 2500 frames over 8hours.
    # Adjust the number of processors (size) and subsample (stride)
    # accordingingly
    
    args = parser.parse_args()
    topfile = args.topfile
    trajfile = args.trajfile
    path_to_ini = args.path_to_ini
    n_frames = args.n_frames
    stride = args.stride
    path_to_py = args.path_to_py
    mpi = args.mpi

    path_to_tables = os.path.dirname(path_to_ini) + "/tables"

    if mpi:
        # If parallel
        from mpi4py import MPI
        comm = MPI.COMM_WORLD   
        size = comm.Get_size()  
        rank = comm.Get_rank()

        if rank == 0:
            prep_minimization(path_to_ini, stride, size=size, path_to_py=path_to_py)

        comm.Barrier()

        os.chdir("inherent_structures")

        # Distribute trajectory chunks to each processor
        all_frame_idxs = np.arange(0, n_frames)
        chunksize = len(all_frame_idxs)/size
        if (len(all_frame_idxs) % size) != 0:
            chunksize += 1
        frames_for_proc = [ all_frame_idxs[i*chunksize:(i + 1)*chunksize:stride] for i in range(size) ]
        n_frames_for_proc = [ len(x) for x in frames_for_proc ]

        if rank == 0:
            # send chunks of trajectory to all other processors. This is meant
            # to limit unneccesary memory usage. Only a chunk is loaded into
            # memory at a time.
            rank_i = 0
            for chunk in md.iterload("../" + trajfile, top="../" + topfile, chunk=chunksize):
                sub_chunk = chunk.slice(np.arange(0, chunk.n_frames, stride))

                if (rank_i == 0) and (rank == 0):
                    traj = sub_chunk
                else:
                    comm.send(sub_chunk, dest=rank_i, tag=11)
                rank_i += 1
            
        frame_idxs = frames_for_proc[rank]
        if rank > 0:
            # receive trajectory chunk.
            traj = comm.recv(source=0, tag=11)

        if not os.path.exists("rank_{}".format(rank)):
            os.mkdir("rank_{}".format(rank))
        os.chdir("rank_{}".format(rank))
        run_minimization(path_to_tables, frame_idxs, traj)
        os.chdir("..")

        # If all trajectories finished then bring them together.
        comm.Barrier()

        if rank == 0:
            frame_idxs = np.concatenate([ np.loadtxt("rank_" + str(x) + "/frames_fin.dat", dtype=int) for x in range(size) ])
            Etot = np.concatenate([ np.loadtxt("rank_" + str(x) + "/Etot.dat") for x in range(size) ])
            np.savetxt("frames_fin.dat", frame_idxs, fmt="%5d")
            np.save("Etot.npy", Etot)

            cat_trajs = " ".join([ "rank_" + str(x) + "/all_frames.xtc" for x in range(size) ])
            with open("trjcat.log", "w") as fout:
                sb.call("trjcat_sbm -f " + cat_trajs + " -o traj.xtc -cat",
                    shell=True, stderr=fout, stdout=fout)

        os.chdir("..")
    else:
        prep_minimization(path_to_ini, stride, path_to_py=path_to_py)
        os.chdir("inherent_structures")

        # run minimization
        frame_idxs = np.arange(0, n_frames, stride)
        traj = md.load("../" + trajfile, top="../" + topfile, stride=stride)
        if not os.path.exists("rank_0"):
            os.mkdir("rank_0")
        os.chdir("rank_0")
        run_minimization(path_to_tables, frame_idxs, traj)
        sb.call("mv all_frames.xtc ../traj.xtc", shell=True)
        sb.call("cp frames_fin.dat ../frames_fin.dat", shell=True)

        os.chdir("..")
        os.chdir("..")

import os
import argparse
import logging
import numpy as np
import subprocess as sb

from mpi4py import MPI

import mdtraj 

import simulation.mdp
import simulation.slurm

import model_builder as mdb

def minimization_script():
    """Script that takes a starting structure and energy minimizes it."""
    script = \
"""#!/bin/bash
# run energy minimization
grompp_sbm -n ../index.ndx -f ../em.mdp -c conf.gro -p ../topol.top -o topol_4.5.tpr &> grompp.log
mdrun_sbm -s topol_4.5.tpr -table ../tables/table.xvg -tablep ../tables/tablep.xvg -tableb ../tables/table &> mdrun.log

# get final energy 
g_energy_sbm -f ener.edr -o Energy -xvg none &> energy.log << EOF
Potential
EOF
tail -n 1 Energy.xvg | awk '{ print $(NF) }' >> Etot.dat

# get final structure 
fstep=`grep "Low-Memory BFGS Minimizer converged" md.log | awk '{ print $(NF-1) }'`
trjconv_sbm -f traj.trr -s topol_4.5.tpr -n ../index.ndx -o temp_frame.xtc -dump ${fstep} &> trjconv.log << EOF
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
"""
    return script

def prep_minimization(model_dir, name):
    """Save model files if needed"""

    # Run parameters
    mdp = simulation.mdp.energy_minimization()
    with open("em.mdp", "w") as fout:
        fout.write(mdp)

    # Load model
    cwd = os.getcwd()
    os.chdir(model_dir)
    model, fitopts = mdb.inputs.load_model(name)
    os.chdir(cwd)

    # Save model files
    model.save_simulation_files(savetables=False)
    if not os.path.exists("tables"):
        os.mkdir("tables")
    os.chdir("tables")
    if not os.path.exists("table.xvg"):
        np.savetxt("table.xvg", model.tablep, fmt="%16.15e", delimiter=" ")
    if not os.path.exists("tablep.xvg"):
        np.savetxt("tablep.xvg", model.tablep, fmt="%16.15e", delimiter=" ")
    for i in range(model.n_tables):
        if not os.path.exists(model.tablenames[i]):
            np.savetxt(model.tablenames[i], model.tables[i], fmt="%16.15e", delimiter=" ")
    os.chdir("..")

def run_minimization(frame_idxs, traj):
    """Perform energy minimization on each frame"""

    n_frames_out = len(frame_idxs)
    
    if os.path.exists("Etot.dat") and (len(np.loadtxt("Etot.dat")) == n_frames_out):
        # Minimization has finished
        pass
    else:
        # Minimization needs to be done
        np.savetxt("frame_idxs.dat", frame_idxs, fmt="%d")
        # Loop over trajectory frames
        for i in xrange(traj.n_frames):
            # perform energy minimization using gromacs
            frm = traj.slice(i)
            frm.save_gro("conf.gro")
            script = minimization_script()
            with open("minimize.bash", "w") as fout:
                fout.write(script)
            cmd = "bash minimize.bash"
            sb.call(cmd.split())

            # record frame idx
            with open("frames_fin.dat", "a") as fout:
                fout.write("{d}\n".format(frame_idxs[i]))

def restart_rank():


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy minimization for inherent structure analysis.")
    parser.add_argument("--name",
                        type=str,
                        required=True,
                        help="Name of .ini file.")

    parser.add_argument("--path_to_ini",
                        type=str,
                        required=True,
                        help="Path to .ini file.")

    parser.add_argument("--stride",
                        type=int,
                        default=10,
                        help="Number of frames to stride. Subsample.")

    parser.add_argument("--n_frames",
                        type=int,
                        default=int(6E5),
                        help="Number of frames in trajectory.")
    
    args = parser.parse_args()
    name = args.name
    model_dir = args.path_to_ini
    n_frames = args.n_frames
    stride = args.stride

    # Performance on one processor is roughly 11sec/frame. 
    # So 1proc can do about 2500 frames over 8hours.
    # Adjust the number of processors (size) and subsample (stride)
    # accordingingly

    #name = "1E0G"
    #model_dir = "/home/ajk8/scratch/6-10-15_nonnative/1E0G/random_b2_0.01/replica_1"
    #model_dir = "/home/ajk8/scratch/6-10-15_nonnative/1E0G/random_b2_1.00/replica_1"
    #n_frames = int(6E5)
    #stride = 10

    comm = MPI.COMM_WORLD   
    size = comm.Get_size()  
    rank = comm.Get_rank()

    if rank == 0:
        if not os.path.exists("inherent_structures"):
            os.mkdir("inherent_structures")
    comm.Barrier()

    os.chdir("inherent_structures")
    prep_minimization(model_dir, name)

    # Distribute trajectory chunks to each processor
    all_frame_idxs = np.arange(0, n_frames)
    chunksize = len(all_frame_idxs)/size
    if (len(all_frame_idxs) % size) != 0:
        chunksize += 1
    frames_for_proc = [ all_frame_idxs[i*chunksize:(i + 1)*chunksize:stride] for i in range(size) ]
    n_frames_for_proc = [ len(x) for x in frames_for_proc ]

    if rank == 0:
        rank_i = 0
        for chunk in mdtraj.iterload("traj.xtc", top="Native.pdb", chunk=chunksize):
            sub_chunk = chunk.slice(np.arange(0, chunk.n_frames, stride))

            if (rank_i == 0) and (rank == 0):
                traj = sub_chunk
            else:
                comm.send(sub_chunk, dest=rank_i, tag=11)
            rank_i += 1
        
    frame_idxs = frames_for_proc[rank]
    if rank > 0:
        traj = comm.recv(source=0, tag=11)
    #print rank, traj.n_frames, traj.time[:2]/0.5, frame_idxs[:2], traj.time[-2:]/0.5, frame_idxs[-2:]  ## DEBUGGING

    if not os.path.exists("rank_{}".format(rank)):
        os.mkdir("rank_{}".format(rank))
    os.chdir("rank_{}".format(rank))
    run_minimization(frame_idxs, traj)
    os.chdir("..")

    # If all trajectories finished then bring them together.

    os.chdir("..")


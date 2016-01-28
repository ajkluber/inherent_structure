import os
import argparse
import numpy as np
import subprocess as sb

import mdtraj 

import simulation.mdp
import simulation.slurm

import model_builder as mdb

def minimization_script(filedir="."):
    """Script that takes a starting structure and energy minimizes it."""
    script = \
"""#!/bin/bash
# run energy minimization
grompp_sbm -n {0}/index.ndx -f {0}/em.mdp -c conf.gro -p {0}/topol.top -o topol_4.5.tpr &> grompp.log
mdrun_sbm -s topol_4.5.tpr -table {0}/tables/table.xvg -tablep {0}/tables/tablep.xvg -tableb {0}/tables/table &> mdrun.log

# get final energy 
g_energy_sbm -f ener.edr -o Energy -xvg none &> energy.log << EOF
Potential
EOF
tail -n 1 Energy.xvg | awk '{{ print $(NF) }}' >> Etot.dat

# get final structure 
fstep=`grep "Low-Memory BFGS Minimizer converged" md.log | awk '{{ print $(NF-1) }}'`
trjconv_sbm -f traj.trr -s topol_4.5.tpr -n {0}/index.ndx -o temp_frame.xtc -dump $fstep &> trjconv.log << EOF
System
EOF

# concatenate to trajectory
if [ ! -e all_frames.xtc ]; then
    mv temp_frame.xtc all_frames.xtc
else
    trjcat_sbm -f all_frames.xtc temp_frame.xtc -o all_frames.xtc -n {0}/index.ndx -nosort -cat &> trjcat.log << EOF
System
EOF
    rm temp_frame.xtc
fi

# cleanup
rm conf.gro mdout.mdp topol_4.5.tpr traj.trr md.log ener.edr confout.gro Energy.xvg
""".format(filedir)
    return script

def prep_minimization(model_dir, name, stride):
    """Save model files if needed"""

    with open("stride", "w") as fout:
        fout.write("{:d}".format(stride))

    # Run parameters
    mdp = simulation.mdp.energy_minimization()
    with open("em.mdp", "w") as fout:
        fout.write(mdp)

    # Load model
    cwd = os.getcwd()
    os.chdir(model_dir)
    print model_dir
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

def run_minimization(frame_idxs, traj, filedir="."):
    """Perform energy minimization on each frame"""
    n_frames_out = len(frame_idxs)

    # Skip frames that have already finished.
    if os.path.exists("frames_fin.dat"):
        frames_fin = np.loadtxt("frames_fin.dat", dtype=int)
        start = len(frames_fin)
        Etot = np.loadtxt("Etot.dat")
        if start < len(Etot):
            np.savetxt("Etot.dat", Etot[:start])
    else:
        start = 0
    
    if start != n_frames_out:
        # Minimization needs to be done
        np.savetxt("frame_idxs.dat", frame_idxs, fmt="%d")

        script = minimization_script(filedir=filedir)
        with open("minimize.bash", "w") as fout:
            fout.write(script)
        cmd = "bash minimize.bash"
        # Loop over trajectory frames
        for i in xrange(start, traj.n_frames):
            # perform energy minimization using gromacs
            frm = traj.slice(i)
            frm.save_gro("conf.gro")
            sb.call(cmd.split())
            # record frame idx
            with open("frames_fin.dat", "a") as fout:
                fout.write("{:d}\n".format(frame_idxs[i]))

def restart(trajfile, topology):
    """TODO: write restart function"""
    # In progress
    frame_idxs = np.loadtxt("frame_idxs.dat", dtype=int)
    frame_fin = np.loadtxt("frames_fin.dat", dtype=int)

    if len(frame_fin) < len(frame_idxs):
        # minimize frames that were not finished
        traj_full = mdtraj.load(trajfile, top=topology)
        traj = traj_full.slice(frame_idxs)

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

    parser.add_argument("--frames_file",
                        type=str,
                        help="Name of file that holds frame.")

    parser.add_argument("--n_frames",
                        type=int,
                        default=int(6E5 + 1),
                        help="Number of frames in trajectory.")

    parser.add_argument("--n_proc",
                        type=int,
                        default=1,
                        help="Number of processors.")
    
    
    args = parser.parse_args()
    name = args.name
    model_dir = args.path_to_ini
    stride = args.stride
    frames_file = args.frames_file
    n_frames = args.n_frames
    n_proc = args.n_proc

    topology = "../Native.pdb"
    trajfile = "../traj.xtc"
    

    # Performance on one processor is roughly 11sec/frame. 
    # So 1proc can do about 2500 frames over 8hours.
    # Adjust the number of processors (size) and subsample (stride)
    # accordingingly

    if n_proc > 1:
        # If parallel
        from mpi4py import MPI
        comm = MPI.COMM_WORLD   
        size = comm.Get_size()  
        rank = comm.Get_rank()

        if rank == 0:
            if not os.path.exists("inherent_structures"):
                os.mkdir("inherent_structures")
            prep_minimization(model_dir, name, stride)
        comm.Barrier()

        os.chdir("inherent_structures")

        traj_whole = mdtraj.load(trajfile, top=topology)
        n_frames = traj_whole.n_frames

        # Distribute trajectory chunks to each processor
        all_frame_idxs = np.arange(0, n_frames)
        chunksize = len(all_frame_idxs)/size
        if (len(all_frame_idxs) % size) != 0:
            chunksize += 1
        frames_for_proc = [ all_frame_idxs[i*chunksize:(i + 1)*chunksize:stride] for i in range(size) ]
        frame_idxs = frames_for_proc[rank]
        n_frames_for_proc = [ len(x) for x in frames_for_proc ]

        traj = traj_whole.slice(frames_for_proc[rank])

#        if rank == 0:
#            print chunksize
#            print frames_for_proc 
#            print size, rank

#        if rank == 0:
#            rank_i = 0
#            tot_frames = 0
#            for chunk in mdtraj.iterload(trajfile, top=topology, chunk=chunksize):
#                sub_chunk = chunk.slice(np.arange(0, chunk.n_frames, stride))
#                tot_frames += chunk.n_frames
#                #print rank_i, tot_frames
#                if (rank_i == 0) and (rank == 0):
#                    traj = sub_chunk
#                else:
#                    print rank_i
#                    comm.send(sub_chunk, dest=rank_i)
#                rank_i += 1
#        rank_i = 0
#        tot_frames = 0
#        for chunk in mdtraj.iterload(trajfile, top=topology, chunk=chunksize):
#            sub_chunk = chunk.slice(np.arange(0, chunk.n_frames, stride))
#            tot_frames += chunk.n_frames
#            if rank_i == rank:
#                print rank_i, rank
#                traj = sub_chunk
#            rank_i += 1
#        print tot_frames
        #if rank == 0:

#        if (rank_i > 0) and (rank_i == rank):
#            print "Received: ", rank_i
#            traj = comm.recv(source=0)

        if 'traj' not in locals():
            print "{} didn't get it".format(rank)
            
        #if rank > 0:
        #    print rank
        #    traj = comm.recv()
        #print rank, traj.n_frames, traj.time[:2]/0.5, frame_idxs[:2], traj.time[-2:]/0.5, frame_idxs[-2:]  ## DEBUGGING

        #print rank
        if not os.path.exists("rank_{}".format(rank)):
            os.mkdir("rank_{}".format(rank))
        os.chdir("rank_{}".format(rank))
        run_minimization(frame_idxs, traj, filedir="..")
        os.chdir("..")
    else:
        if not os.path.exists("inherent_structures"):
            os.mkdir("inherent_structures")
        os.chdir("inherent_structures")
        prep_minimization(model_dir, name, stride)
        frame_idxs = np.arange(0, n_frames, stride)
        traj = mdtraj.load(trajfile, top=topology, stride=stride)
        run_minimization(frame_idxs, traj, filedir=".")

    os.chdir("..")

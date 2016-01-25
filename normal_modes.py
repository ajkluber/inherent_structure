import os
import argparse
import numpy as np
import subprocess as sb

import mdtraj 

import simulation.mdp
import simulation.slurm

import model_builder as mdb

def normal_mode_script(idx, filedir="."):
    script = \
"""#!/bin/bash
grompp_sbm -n {0}/index.ndx -f hessian.mdp -c conf.gro -p {0}/topol.top -o topol.tpr &> grommp.log
mdrun_sbm -mtx -s topol.tpr -table {0}/tables/table.xvg -tablep {0}/tables/tablep.xvg -tableb {0}/tables/table  &> mdrun.log
g_nmeig_sbm -f nm.mtx -s topol.tpr -ol vals -xvg none  &> g_nmeig.log
awk '{{ print ($NF) }}' vals.xvg > vals_{1:d}.dat

# cleanup
rm mdout.mdp topol.tpr nm.mtx md.log eigenval.xvg vals.xvg eigenfreq.xvg eigenvec.trr
""".format(filedir, idx)
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
    topology = "../Native.pdb"
    trajfile = "all_frames.xtc"

    traj = mdtraj.load(trajfile, top=topology)

    filedir = "../.."

    if not os.path.exists("nma"):
        os.mkdir("nma")
    os.chdir("nma")

    mdp = simulation.mdp.normal_modes()
    with open("hessian.mdp", "w") as fout:
        fout.write(mdp)

    #TODO: concatenate eigenvals?

    cmd = "bash normal.bash"
    for i in range(traj.n_frames):
        print i
        frm = traj.slice(i)
        frm.save_gro("conf.gro")
        script = normal_mode_script(i, filedir=filedir)
        with open("normal.bash", "w") as fout:
            fout.write(script)
        sb.call(cmd.split())

        #with open("frames_fin.dat", "a") as fout:
        #    fout.write("{:d}\n".format(frame_idxs[i]))




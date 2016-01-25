import os
import argparse
import numpy as np
import subprocess as sb

import mdtraj 

import simulation.mdp
import simulation.slurm

import model_builder as mdb

def normal_mode_script(filedir="."):
    script = \
"""#!/bin/bash
grompp_sbm -n {0}/index.ndx -f hessian.mdp -c conf.gro -p {0}/topol.top -o topol.tpr
mdrun_sbm -mtx -s topol.tpr -table {0}/tables/table.xvg -tablep {0}/tables/tablep.xvg -tableb {0}/tables/table
g_nmeig_sbm -f nm.mtx -s topol.tpr -ol vals -xvg none
awk '{{ print ($NF) }}' vals.xvg > vals.dat

# cleanup
rm mdout.mdp topol.tpr nm.mtx md.log eigenval.xvg vals.xvg eigenfreq.xvg eigenvec.trr
""".format(filedir)



  
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
    ## NOT DONE YET.
    topology = "../Native.pdb"
    trajfile = "../traj.xtc"

    traj = mdtraj.load(trajfile, top=topology)


# inherent_structure

Inherent structure analysis was derived by Stillinger and Weber as a way to
approximate the features of an energy landscape[1].

Calculate "inherent structures".



### Usage

Generating the inherent structures is the most computationally demanding step.
The process requires 1) generating an equilibrated trajectory at constant
temperature, then 2) performing energy minimization on that trajectory. This
package contains scripts to do energy minimization with Gromacs.

`srun -n 12 python generate.py --path_to_ini SH3.ini --stride 10 --n_frames 100000 --topfile ref.pdb`


[1]: Stillinger, F. H.; Weber, T. A. Hidden Structure in Liquids. Phys. Rev. A 1982, 25, 978-989.

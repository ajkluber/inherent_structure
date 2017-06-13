import numpy as np

def get_frame_idxs(temp_dirs):
    frame_idxs = np.concatenate([ np.loadtxt(x + "/inherent_structures/frames_fin.dat", dtype=int) for x in temp_dirs ])
    return frame_idxs

def get_total_energy(temp_dirs):
    Etot = np.concatenate([ np.load(x + "/inherent_structures/Etot.npy") for x in temp_dirs ])
    return Etot

def get_backbone_energy(temp_dirs):
    Eback = np.concatenate([ np.load(x + "/inherent_structures/Ebackbone.npy") for x in temp_dirs ])
    return Eback

def get_native_nonnative_energies(temp_dirs):
    """Get any energies that exist from subdirectories"""

    Enat = np.concatenate([ np.load(x + "/inherent_structures/Enat.npy") for x in temp_dirs ])
    Enon = np.concatenate([ np.load(x + "/inherent_structures/Enon.npy") for x in temp_dirs ])
    return Enat, Enon


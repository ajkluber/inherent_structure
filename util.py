import os
import glob
import numpy as np

def get_T_used():
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        Tf = float(fin.read())
    return Tf

def get_frame_idxs():
    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    frame_idxs = np.concatenate([ np.loadtxt(x + "/inherent_structures/frames_fin.dat", dtype=int) for x in temp_dirs ])
    return frame_idxs

def get_load_func(filename):
    if filename.endswith("npy"):
        load_func = np.load
    elif filename.endswith("dat"):
        load_func = np.loadtxt
    else:
        raise IOError("Enrecognized filetype: {}".format(filename))
    return load_func

def get_energies(filenames, use_rank=True):
    """Get any energies that exist from subdirectories
    
    Parameters
    ----------
    filenames : list
        List of filenames.
    use_rank : bool, opt.
        Use energy in rank_* subdirectories
    """

    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    cwd = os.getcwd()
    E = [ [] for x in range(len(filenames)) ]
    for i in range(len(temp_dirs)):
        os.chdir(temp_dirs[i] + "/inherent_structures")
        if use_rank: 
            # get energies in rank subdirectory
            size = len(glob.glob("rank_*"))
            for j in range(len(filenames)):
                load_func = get_load_func(filenames[j])
                E_temp = np.concatenate([ load_func("rank_{}/{}".format(x, filenames[j])) for x in range(size) ])
                E[j].append(E_temp)
        else:
            # get energy in parent directory
            for j in range(len(filenames)):
                load_func = get_load_func(filenames[j])
                E_temp = load_func(filenames[j])
                E[j].append(E_temp)

        os.chdir(cwd)
    Ecat = [ np.concatenate(E[i]) for i in range(len(filenames)) ]
    return Ecat

def get_total_energy():
    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    cwd = os.getcwd()
    Etot = []
    for i in range(len(temp_dirs)):
        os.chdir(temp_dirs[i] + "/inherent_structures")
        size = len(glob.glob("rank_*/Etot.dat"))
        Etot_temp = np.concatenate([ np.loadtxt("rank_{}/Etot.dat".format(x)) for x in range(size) ])
        Etot.append(Etot_temp)
        os.chdir(cwd)
    Etot = np.concatenate(Etot)
    #Etot = np.concatenate([ np.load(x + "/inherent_structures/Etot.npy") for x in temp_dirs ])
    return Etot

def get_backbone_energy():
    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    cwd = os.getcwd()
    Ebackbone = []
    for i in range(len(temp_dirs)):
        os.chdir(temp_dirs[i] + "/inherent_structures")
        size = len(glob.glob("rank_*/Ebackbone.npy"))
        Ebackbone_temp = np.concatenate([ np.load("rank_{}/Ebackbone.npy".format(x)) for x in range(size) ])
        Ebackbone.append(Ebackbone_temp)
        os.chdir(cwd)
    Ebackbone = np.concatenate(Ebackbone)
    #Eback = np.concatenate([ np.load(x + "/inherent_structures/Ebackbone.npy") for x in temp_dirs ])
    return Eback


def get_native_nonnative_energies():
    """Get any energies that exist from subdirectories"""

    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    cwd = os.getcwd()
    Enat = []
    Enon = []
    for i in range(len(temp_dirs)):
        os.chdir(temp_dirs[i] + "/inherent_structures")
        size = len(glob.glob("rank_*/Enon.npy"))
        Enat_temp = np.concatenate([ np.load("rank_{}/Enat.npy".format(x)) for x in range(size) ])
        Enon_temp = np.concatenate([ np.load("rank_{}/Enon.npy".format(x)) for x in range(size) ])
        Enat.append(Enat_temp)
        Enon.append(Enon_temp)
        os.chdir(cwd)
    Enat = np.concatenate(Enat)
    Enon = np.concatenate(Enon)

    #Enat = np.concatenate([ np.load(x + "/inherent_structures/Enat.npy") for x in temp_dirs ])
    #Enon = np.concatenate([ np.load(x + "/inherent_structures/Enon.npy") for x in temp_dirs ])
    return Enat, Enon

def get_thermal_native_nonnative_energies():
    """Get any energies that exist from subdirectories"""

    Tf = util.get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    Enat = np.concatenate([ np.load(x + "/inherent_structures/Enat_thm.npy") for x in temp_dirs ])
    Enon = np.concatenate([ np.load(x + "/inherent_structures/Enon_thm.npy") for x in temp_dirs ])
    return Enat, Enon

def determine_U_frames(Enat, nbins_Enat):

    P_Enat, Enat_bins = np.histogram(Enat, bins=nbins_Enat)
    Enat_mid_bin = 0.5*(Enat_bins[1:] + Enat_bins[:-1])

    # max P_Enat above and below median
    peak_idx1 = np.argwhere(P_Enat == np.max(P_Enat[Enat_mid_bin < np.median(Enat)]))[0][0]
    peak_idx2 = np.argwhere(P_Enat == np.max(P_Enat[Enat_mid_bin >= np.median(Enat)]))[0][0]

    # U = an interval around the unfolded state peak in P_Enat
    # Determine the width of the non-native interaction basin -> roughness
    left_side = False
    right_side = False
    threshold = 0.6
    for i in range(len(Enat_mid_bin)):
        if Enat_mid_bin[i] >= np.median(Enat):
            if not left_side:
                if P_Enat[i] >= threshold*P_Enat[peak_idx2]:
                    left_side = i
            if left_side:
                if P_Enat[i] < threshold*P_Enat[peak_idx2]:
                    right_side = i
                    break

    U = (Enat > Enat_mid_bin[left_side]) & (Enat <= Enat_mid_bin[right_side])
    return P_Enat, Enat_mid_bin, U, peak_idx1, peak_idx2


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

def get_data(filenames, use_rank=True):
    """Get data from IS calculations
    
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
        for j in range(len(filenames)):
            ranks_exist = len(glob.glob("rank_*/{}".format(filenames[j]))) > 0
            if use_rank and ranks_exist: 
                # get data from rank subdirectory
                size = len(glob.glob("rank_*"))
                load_func = get_load_func(filenames[j])
                E_temp = np.concatenate([ load_func("rank_{}/{}".format(x, filenames[j])) for x in range(size) ])
                E[j].append(E_temp)
            else:
                # get data from parent directory
                for j in range(len(filenames)):
                    load_func = get_load_func(filenames[j])
                    E_temp = load_func(filenames[j])
                    E[j].append(E_temp)

        os.chdir(cwd)
    Ecat = [ np.concatenate(E[i]) for i in range(len(filenames)) ]
    return Ecat

def determine_U_frames(Enat, nbins_Enat, threshold=0.5):

    P_Enat, Enat_bins = np.histogram(Enat, bins=nbins_Enat)
    Enat_mid_bin = 0.5*(Enat_bins[1:] + Enat_bins[:-1])

    # max P_Enat above and below median
    peak_idx1 = np.argwhere(P_Enat == np.max(P_Enat[Enat_mid_bin < np.median(Enat_mid_bin)]))[0][0]
    peak_idx2 = np.argwhere(P_Enat == np.max(P_Enat[Enat_mid_bin >= np.median(Enat_mid_bin)]))[0][0]

    # U = an interval around the unfolded state peak in P_Enat
    # Determine the width of the non-native interaction basin -> roughness
    left_side = None
    right_side = None
    for i in range(len(Enat_mid_bin)):
        # Unfolded state is the higher energy state
        if Enat_mid_bin[i] >= np.median(Enat_mid_bin):
            if left_side is None:
                if P_Enat[i] >= threshold*P_Enat[peak_idx2]:
                    left_side = i
            if not (left_side is None) and (i > (left_side + 4)):
                # find right side only after left side is found
                if P_Enat[i] < threshold*P_Enat[peak_idx2]:
                    right_side = i
                    break

    U = (Enat > Enat_mid_bin[left_side]) & (Enat <= Enat_mid_bin[right_side])
    return P_Enat, Enat_mid_bin, U, peak_idx1, peak_idx2, left_side, right_side


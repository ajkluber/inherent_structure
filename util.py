import numpy as np

def get_frame_idxs():
    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    frame_idxs = np.concatenate([ np.loadtxt(x + "/inherent_structures/frames_fin.dat", dtype=int) for x in temp_dirs ])
    return frame_idxs

def get_total_energy():
    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    Etot = np.concatenate([ np.load(x + "/inherent_structures/Etot.npy") for x in temp_dirs ])
    return Etot

def get_backbone_energy():
    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    Eback = np.concatenate([ np.load(x + "/inherent_structures/Ebackbone.npy") for x in temp_dirs ])
    return Eback

def get_native_nonnative_energies():
    """Get any energies that exist from subdirectories"""

    Tf = get_T_used()
    temp_dirs = [ "T_{:.2f}_{}".format(Tf,x) for x in [1,2,3] ]
    Enat = np.concatenate([ np.load(x + "/inherent_structures/Enat.npy") for x in temp_dirs ])
    Enon = np.concatenate([ np.load(x + "/inherent_structures/Enon.npy") for x in temp_dirs ])
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
    return P_Enat, Enat_mid_bin, U

def get_T_used():
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        Tf = float(fin.read())
    return Tf

import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    n_frames = 218
    E = np.loadtxt("../Etot.dat")

    skip = 7
    ground_state = np.loadtxt("vals_0.dat")[skip:]
    
    T = 120.7
    kb = 0.0083145
    

    Fvib = []
    for i in range(1, n_frames):
        print i
        vib_temp = np.loadtxt("vals_{:d}.dat".format(i))[skip:]
        # skip modes that are 0.0
        nonzero = vib_temp > 0
        Fvib_temp = 0.5*kb*T*np.sum(np.log(vib_temp[nonzero]/ground_state[nonzero]))
        Fvib.append(Fvib_temp)
        
    #np.loadtxt("vals_{:d}.dat".format(x), usecols=(1,))
    plt.figure()
    plt.plot(E[1:], Fvib, 'o')
    plt.savefig("E_vs_Fvib.png")
    plt.show()

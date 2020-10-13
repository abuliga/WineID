# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:01:49 2020

@author: Paolo
"""

import tensorly as tl
from tensorly.decomposition import parafac2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)), dtype=tl.float64)
unfolded = tl.unfold(tensor, mode=0)
tl.fold(unfolded, mode=0, shape=tensor.shape)



num_spec = 2; #number of specimens
num_samples = 5; #number of chromatograms

#define and plot spectres
freqs = 10;
spectres = [np.random.random_sample(freqs) for i in range(num_spec)]
plt.figure();
for spect in spectres:
    plt.plot(range(freqs),spect);
plt.title("spectres")
plt.show()


#define elution profiles
def gaussian(times, peak, sigma): 
    return np.array([np.exp((-(x - peak)**2)/sigma) for x in times])

time_steps = 100;
times = np.linspace(0,1,time_steps);

peaks_per_profile = 2;
peaks = [[np.random.random_sample(peaks_per_profile) for j in range(num_spec)] for i in range(num_samples)] #each elution profile has multiple peaks. One profile per specimens, per sample.
sigmas = [[0.01 + np.random.random_sample(peaks_per_profile)/100 for j in range(num_spec)] for i in range(num_samples)]


#make chromas
chromatograms = [];
for i in range(num_samples):
    #initialize single chromatogram
    chrome = np.tensordot(np.zeros(freqs),np.zeros(time_steps),0)
    
    for j in range(num_spec):
        #loop through specimens
        ps = peaks[i][j];
        ss = sigmas[i][j];
        
        elution_profile = np.zeros(time_steps);
        for p,s in zip(ps,ss):
            #find elution profile (depends on specimens and chromatogram)
            elution_profile += gaussian(times, p,s);
        chrome += np.tensordot(spectres[j],elution_profile,0)
    chromatograms.append(tl.tensor(chrome,dtype=tl.float64));



for i in range(num_samples):
    fig = plt.figure();
    ax = fig.gca(projection='3d')
    X = times;
    Y = range(freqs);
    X, Y = np.meshgrid(X, Y)
    
    Z = chromatograms[i];
    
    surf = ax.plot_surface(X, Y, Z,
                           linewidth=0, antialiased=False)


parafac2(chromatograms,2)



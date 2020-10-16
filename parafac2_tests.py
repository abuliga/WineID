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
import numpy.linalg as la


tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)), dtype=tl.float64)
unfolded = tl.unfold(tensor, mode=0)
tl.fold(unfolded, mode=0, shape=tensor.shape)



num_spec = 2; #number of specimens
num_samples = 10; #number of chromatograms

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
    
    #surf = ax.plot_surface(X, Y, Z,
    #                      linewidth=0, antialiased=False)

best_err = np.inf
decomposition = None

for run in range(10):
    print(f'Training model {run}...')
    trial_decomposition, trial_errs = parafac2(chromatograms, num_samples, return_errors=True, tol=1e-8, n_iter_max=500, random_state=run)
    print(f'Number of iterations: {len(trial_errs)}')
    print(f'Final error: {trial_errs[-1]}')
    if best_err > trial_errs[-1]:
        best_err = trial_errs[-1]
        err = trial_errs
        decomposition = trial_decomposition
    print('-------------------------------')
print(f'Best model error: {best_err}')

est_tensor = tl.parafac2_tensor.parafac2_to_tensor(decomposition)
est_weights, (est_A, est_B, est_C) = tl.parafac2_tensor.apply_parafac2_projections(decomposition)
est_A, est_projected_Bs, est_C = tl.parafac2_tensor.apply_parafac2_projections(decomposition)[1]
sign = np.sign(est_A)
est_A = np.abs(est_A)
est_projected_Bs = sign[:, np.newaxis]*est_projected_Bs


est_A_normalised = est_A/la.norm(est_A, axis=0)
est_Bs_normalised = [est_B/la.norm(est_B, axis=0) for est_B in est_projected_Bs]
est_C_normalised = est_C/la.norm(est_C, axis=0)



fig, axes = plt.subplots(num_spec, 3, figsize=(15, 3*num_spec+1))

for r in range(1):
    
    # Plot true and estimated components for mode A
    axes[r][0].plot(est_A_normalised,'--', label='Estimated')
    
    # Labels for the different components
    axes[r][0].set_ylabel(f'Component {r}')

    # Plot true and estimated components for mode C
    axes[r][2].plot(est_C_normalised, '--')

    # Get the signs so that we can flip the B mode factor matrices
    A_sign = np.sign(est_A_normalised)


print("meow")
plt.figure()


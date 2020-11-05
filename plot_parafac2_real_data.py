# -*- coding: utf-8 -*-
"""
Demonstration of PARAFAC2 with real data
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import tensorly as tl
import os
from tensorly.decomposition import parafac2
import pickle as pk
import math
from scipy.optimize import linear_sum_assignment
from io import StringIO  

exp_info_names =["Interval","Start Time","End Time", "Start Wavelength", "End Wavelength","Wavelength Axis Points", "Time Axis Points"];

folder = "my_folder";


def import_chromatogram_from_txt(experiment, filename, frequencies = [], start_time = 0, end_time = math.inf):
    temp_data = [];
    experiment["info"] = {};
    experiment["info"]["filename"] = filename;
    with open(filename) as f:
        #search for 3D data
        while (f.readline().find("[PDA 3D]") == -1):
            1;
        
        #collect infos
        for info in exp_info_names:
            experiment["info"][info] = float(f.readline().split(",")[1]); 
        f.readline()
        f.readline()
        
        time = experiment["info"]["Start Time"];
        end_time = min(experiment["info"]["End Time"], end_time);
        increment = (end_time - time)/experiment["info"]["Time Axis Points"];
        while(True):
            line = f.readline()
            #print(time)
            #collect experiment results
            try:
                c = [float(elem) for elem in line.split(",")]
                if frequencies:
                    c = np.array(c)[frequencies]

                #collect data only if in desired time interval
                if (time >= start_time):
                    if (time < end_time):
                        temp_data.append(c);
                    else:
                        break
                

                time += increment;
            except ValueError:
                break
            
            #find and collect experiment infos
            if (line.find("[PDA 3D]") != -1):
                for info in exp_info_names:
                    experiment["info"][info] = float(f.readline().split(",")[1]); 
                f.readline()
                f.readline()
    experiment["data"] = np.array(temp_data);
    print("Experiment loaded. INFO: ",experiment["info"])
        
files = [];
for r, d, f in os.walk(folder):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file));
            
##############################################################################
            #IMPORT PHASE
##############################################################################
experiments = [];
times = [];
tensor = [];
freqs = [];
for file in files:
    experiment = {};
    import_chromatogram_from_txt(experiment, file, freqs, start_time = 34, end_time = 40);
    print(experiment["data"].shape)
    ratio = 5
    experiment["data"] = experiment["data"][::ratio][:]
    experiment["data"] = experiment["data"]
    print(experiment["data"].shape)
    tensor.append(experiment["data"])


##############################################################################
# Fit a PARAFAC2 tensor
# ---------------------
# To avoid local minima, we initialise and fit multiple models and choose the one
# with the lowest error


best_err = np.inf
decomposition = None
ranks = range(2,8)
number_of_runs = 5;

for run in range(number_of_runs):
    for rank in ranks:
        print(f'Training model {run}...')
        print(f'Testing rank {rank}')
        trial_decomposition, trial_errs = parafac2(tensor, rank, return_errors=True, tol=1e-8, normalize_factors = True, n_iter_max=500, random_state=run)
        print(f'Number of iterations: {len(trial_errs)}')
        print(f'Final error: {trial_errs[-1]}')
        if best_err > trial_errs[-1]:
            best_err = trial_errs[-1]
            err = trial_errs
            decomposition = trial_decomposition
            true_rank = rank;
        print('-------------------------------')
print(f'Best model error: {best_err} with rank {true_rank}')

est_tensor = tl.parafac2_tensor.parafac2_to_tensor(decomposition)
est_weights, (est_A, est_B, est_C) = tl.parafac2_tensor.apply_parafac2_projections(decomposition)

##############################################################################
# Compute performance metrics
# ---------------------------


reconstruction_error = la.norm(est_tensor - tensor)
recovery_rate = 1 - reconstruction_error/la.norm(tensor)

print(f'{recovery_rate:2.0%} of the data is explained by the model')

##############################################################################
# Visualize the components
# ------------------------
est_A, est_projected_Bs, est_C = tl.parafac2_tensor.apply_parafac2_projections(decomposition)[1]
sign = np.sign(est_A)
est_A = np.abs(est_A)
est_projected_Bs = sign[:, np.newaxis]*est_projected_Bs

est_A_normalised = est_A/la.norm(est_A, axis=0)
est_Bs_normalised = [est_B/la.norm(est_B, axis=0) for est_B in est_projected_Bs]
est_C_normalised = est_C/la.norm(est_C, axis=0)

# Create plots of each component vector for each mode
# (We just look at one of the B_i matrices)

fig, axes = plt.subplots(true_rank, 3, figsize=(15, 3*true_rank+1))
i = 0 # What slice, B_i, we look at for the B mode

for r in range(true_rank):
    
    # Plot true and estimated components for mode A
    axes[r][0].plot((est_A_normalised[:, r]),'--', label='Estimated')
    
    # Labels for the different components
    axes[r][0].set_ylabel(f'Component {r}')

    # Plot true and estimated components for mode C
    axes[r][2].plot(est_C_normalised[:, r], '--')

    A_sign = np.sign(est_A_normalised)
    # Plot estimated components for mode B (after sign correction)
    axes[r][1].plot(A_sign[i, r]*est_Bs_normalised[i][:, r], '--')

# Titles for the different modes
axes[0][0].set_title('Concentration')
axes[0][2].set_title('Spectra')
axes[0][1].set_title(f'Elution profile (slice {i})')

# Create a legend for the entire figure  
handles, labels =  axes[r][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

##############################################################################
# Inspect the convergence rate
# ----------------------------

loss_fig, loss_ax = plt.subplots(figsize=(9, 9/1.6))
loss_ax.plot(range(1, len(err)), err[1:])
loss_ax.set_xlabel('Iteration number')
loss_ax.set_ylabel('Relative reconstruction error')
mathematical_expression_of_loss = r"$\frac{\left|\left|\hat{\mathcal{X}}\right|\right|_F}{\left|\left|\mathcal{X}\right|\right|_F}$"
loss_ax.set_title(f'Loss plot: {mathematical_expression_of_loss} \n (starting after first iteration)', fontsize=16)
xticks = loss_ax.get_xticks()
loss_ax.set_xticks([1] + list(xticks[1:]))
loss_ax.set_xlim(1, len(err))
plt.tight_layout()
plt.show()



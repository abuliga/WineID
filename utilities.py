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
import math
import joblib as jlb
import pickle as pk
import glob as glb




exp_info_names =["Interval","Start Time","End Time", "Start Wavelength", "End Wavelength","Wavelength Axis Points", "Time Axis Points"];

    


class chromatogram():

    def __init__(self):
        self.info = {};
        self.data = [];
        self.current_start_time = 0;
        self.current_end_time = 0;
        self.current_time_resolution = 1;
    
    def restore_default(self):
        self.current_start_time = self.info["Start Time"];
        self.current_end_time = self.info["End Time"];
        self.current_time_resolution = 1;
        
    def set_start_time(self, start):
        if(start >= self.info["Start Time"] and start < self.current_end_time):    
            self.current_start_time = start;
        else:
            print("WARNING: trying to set an invalid start time");
    
    def set_end_time(self, end):
        if(end <= self.info["End Time"] and end > self.current_start_time):   
            self.current_end_time = end;
        else:
            print("WARNING: trying to set an invalid end time");
    
    def set_start_idx(self, idx):
        new_time = self.idx_to_sec(idx);
        print(new_time);
        self.set_start_time(new_time);
        
    def set_end_idx(self, idx):
        new_time = self.idx_to_sec(idx);
        print(new_time);
        self.set_end_time(new_time);
    
    def cut(self):
        start = self.sec_to_indx(self.current_start_time);
        end = self.sec_to_indx(self.current_end_time);
        self.data = self.data[start:end:self.current_time_resolution, :]
    
    def n_time(self):
        return np.shape(self.data)[0];
    def n_freqs(self):
        return np.shape(self.data)[1];
    
    def time_step(self):
        return (self.info["End Time"] - self.info["Start Time"])/self.info["Time Axis Points"];
    
    def sec_to_indx(self, second):
        if second <= self.info["End Time"]:
            return int((second - self.info["Start Time"])/self.time_step());
        else:
            print("WARNING: chromatogram ends before the requested time; returning end time");
            return self.info["Time Axis Points"];
        
    def idx_to_sec(self, idx):
        if idx <= self.info["Time Axis Points"]:
            return self.info["Start Time"] + idx*self.time_step();
        else:
            print("invalid index");
            return 0;
        
    def get_data(self, frequencies = []):
        
        start = self.sec_to_indx(self.current_start_time);
        end = self.sec_to_indx(self.current_end_time);
        res = self.current_time_resolution;
        print(start,res,end);
        if(frequencies):
            return self.data[start:end:res, frequencies];
        else:
            return self.data[start:end:res, :];
        
class experiment():
        
    def __init__(self, freqs = []):
        self.chromes = [];
        self.chromenames = [];
        self.frequencies = freqs.copy();
        self.peaks = [];
        
    def n_chromes(self):
        return len(self.chromes)
    
    def n_freqs(self):
        return (self.chromes[0]).n_freqs();
    
    def add_freq(self, freq):
        if not(freq in self.frequencies):
            self.frequencies.append(freq);
    
    def add_chrome(self, chrome):
        if not(chrome.info["chromename"] in self.chromenames):
            self.chromes.append(chrome);
            self.chromenames.append(chrome.info["chromename"])
        else:
            print("WARNING: This experiment already contains a chromatogram with this name")
        
    def tensor(self, freqs):
        return [chrome.get_data(freqs) for chrome in self.chromes];
    
def files_in_dir(folder):
    files = [];
    for r, d, f in os.walk(folder):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file));
    return files;



def import_folder(folder_name, new_folder, start = 0, end = math.inf, timestep = 1):
    
    try:
        os.mkdir(new_folder);
    except:
        True;
    
    chrome_files = glb.glob(folder_name + "/*.txt");
    for filename in chrome_files:
        shortname = filename.split("\\")[-1];
        shortname = shortname.split(".")[0];
        chrome = import_chromatogram_from_txt(filename, start_time = start, end_time = end, timestep = timestep)
        outfile_name = new_folder + "/" + shortname + ".joblib";
        jlb.dump(chrome.data, outfile_name);


    
    
def import_chromatogram_from_txt(filename, frequencies = [], start_time = 0, end_time = math.inf, timestep = 1):
    chrome = chromatogram();
    temp_data = [];
    chromename = filename.split("\\")[-1];
    chromename = chromename.split(".")[0];
    chrome.info["filename"] = filename;
    chrome.info["chromename"] = chromename;
    with open(filename) as f:
        #search for 3D data
        line = ""
        while (line.find("[PDA 3D]") == -1):
            line = f.readline();
    
        #collect infos
        for info in exp_info_names:
            chrome.info[info] = float(f.readline().split(",")[1]); 
        f.readline()
        f.readline()
        
        time = chrome.info["Start Time"];
        end_time = min(chrome.info["End Time"], end_time);
        increment = (chrome.info["End Time"] - chrome.info["Start Time"])/chrome.info["Time Axis Points"];
        
        idx = -1;
        while(True):
            line = f.readline()

            try: 
                #time controls
                time += increment;
                idx += 1;
                if(time < start_time):
                    continue;
                if(time >= end_time):
                    break;
                if(idx%timestep != 0):
                    continue;
                    
                #data import
                c = [float(elem) for elem in line.split(",")]
                if frequencies:
                    c = np.array(c)[frequencies];

                temp_data.append(c);
                   
            except ValueError:
                break
            

    chrome.info["Start Time"] = max(start_time, chrome.info["Start Time"]);
    chrome.info["End Time"] = end_time;
    chrome.current_start_time = chrome.info["Start Time"];
    chrome.current_end_time = chrome.info["End Time"];
    chrome.current_time_resolution = 1;
    chrome.data = np.array(temp_data);
    print(chrome.data.shape)
    chrome.info["Time Axis Points"] = chrome.data.shape[0];

    print("chromatogram loaded from file: ", chrome.info["filename"])
    return chrome;        


##############################################################################
# Fit a PARAFAC2 tensor
# ---------------------
# To avoid local minima, we initialise and fit multiple models and choose the one
# with the lowest error

def decompose (tensor):
    
    best_err = np.inf
    decomposition = None
    rank = 2
    number_of_runs = 10;
    
    for run in range(number_of_runs):
        print(f'Training model {run}...')
        print(f'Testing rank {rank}')
        trial_decomposition, trial_errs = parafac2(tensor, rank, return_errors=True, tol=1e-9, normalize_factors = True, n_iter_max=300, random_state=run)
        print(f'Number of iterations: {len(trial_errs)}')
        print(f'Final error: {trial_errs[-1]}')
        if best_err > trial_errs[-1]:
            best_err = trial_errs[-1]
            err = trial_errs
            decomposition = trial_decomposition
        print('-------------------------------')
    print(f'Best model error: {best_err} with rank {rank}')
    
    #est_tensor = tl.parafac2_tensor.parafac2_to_tensor(decomposition)    
    
    ##############################################################################
    # Compute performance metrics# ---------------------------

    return decomposition, rank, err

##############################################################################
# Visualize the components
# ------------------------
def plot_decomposition(decomposition, true_rank, err):
    est_A, est_projected_Bs, est_C = tl.parafac2_tensor.apply_parafac2_projections(decomposition)[1]
    sign = np.sign(est_A)
    est_A = np.abs(est_A)
#    est_projected_Bs = sign[:, np.newaxis]*est_projected_Bs
    
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



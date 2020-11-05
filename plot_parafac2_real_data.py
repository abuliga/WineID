# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:41:53 2020

@author: Paolo
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import tensorly as tl
import os
from tensorly.decomposition import parafac2
import utilities as utl
import wineUI as ui
import math

folder = "C:/Users/Paolo/OneDrive/Desktop/seq_1/"; 

files = utl.files_in_dir(folder)
print("files found:", files)
##############################################################################
            #IMPORT PHASE
##############################################################################

freqs = [];
timestep = 3
exp = utl.experiment(freqs);
for file in files:
    chrome = utl.chromatogram();
    utl.import_chromatogram_from_txt(chrome, file, freqs, start_time = 0, end_time = 20, timestep = timestep);
    exp.add_chrome(chrome);
    #ui.show_chromatogram(chrome);
    
decomposition, true_rank, error = utl.decompose(exp.tensor());
utl.plot_decomposition(decomposition, true_rank, error)
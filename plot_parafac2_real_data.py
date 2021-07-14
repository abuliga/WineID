# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:41:53 2020

@author: Paolo
"""
#!pip install tensorly
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import tensorly as tl
import os
from tensorly.decomposition import parafac2
import utilities as utl
import wineUI as ui
import math

folder = "./data/"; 

files = utl.files_in_dir(folder)
print("files found:", files)
##############################################################################
            #IMPORT PHASE
##############################################################################

freqs = [50,100,150]
print(freqs)
timestep = 3
exp = utl.experiment(freqs);

for idx, file in enumerate(files):
    print("importing", file)
    chrome = utl.import_chromatogram_from_txt(file, freqs, start_time = 20, end_time = 25, timestep = 5);
    print(chrome.get_data().shape);
    exp.add_chrome(chrome);

ui.experiment_UI(exp);

decomposition, true_rank, error = utl.decompose(exp.tensor([]));
utl.plot_decomposition(decomposition, true_rank, error)
utl.create_columns(decomposition,freqs,true_rank)
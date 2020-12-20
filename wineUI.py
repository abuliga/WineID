# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:56:04 2020

@author: Paolo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import utilities as utl


class adjustable_plot():

    def __init__(self, chrome, fig, grid, idx, widgets):
        self.idx = idx;
        self.chrome = chrome
        self.fig = fig;
        self.ax = plt.subplot(grid[widgets*idx:widgets*(idx+1):, 0]);
        self.ax.margins(x=0)
        
        freqs = chrome.n_freqs();
        timepoints = chrome.n_time();
        
        default_freq = 0;
        default_start = 0;
        default_end = timepoints;
        
        y = get_freq_slice(chrome,default_freq,default_start, default_end);
        x = range(timepoints);
        ylim = np.max(chrome.data);
        
        self.plot, = self.ax.plot(x, y)
        self.ax.set_ylim(0, ylim)

        #set sliders
        axfreq = plt.subplot(grid[widgets*idx + 1, 1])
        axstart = plt.subplot(grid[widgets*idx + 2, 1])
        axend = plt.subplot(grid[widgets*idx + 3, 1])
        resetax = plt.subplot(grid[widgets*idx + 0, 1])

        #set sliders
        self.freq_slider = Slider(axfreq, 'Freq', 0, freqs, valinit=default_freq)
        self.start_slider = Slider(axstart, 'start', 0, len(x), valinit=default_start)
        self.end_slider = Slider(axend, 'end', 0, timepoints , valinit=default_end)
        self.button = Button(resetax, 'Reset', hovercolor='0.975')
        
        self.freq_slider.on_changed(self.update)
        self.start_slider.on_changed(self.update)
        self.end_slider.on_changed(self.update)
        self.button.on_clicked(self.reset)

        
    def update(self, val):
        start = int(self.start_slider.val)
        end = int(self.end_slider.val)
        freq = int(self.freq_slider.val)
        new_y = get_freq_slice(self.chrome, freq, start, end);
        new_x = range(end - start);
        self.plot.set_data(new_x, new_y)
        self.ax.relim()
        self.ax.autoscale_view(True,True,True)
        self.fig.canvas.draw_idle()     
    
    def reset(self, event):
            self.start_slider.reset();
            self.end_slider.reset();
            self.freq_slider.reset();

def get_freq_slice(chrome, freq, start, end):
    return chrome.data[start:end, freq];

def experiment_UI(exp):
    fig = plt.figure();
    
    n_chromes = exp.n_chromes();
    widgets = 4;
    grid = plt.GridSpec(n_chromes*widgets, 2, wspace = .25, hspace = .20)

    plots = []
    for idx, chrome in enumerate(exp.chromes):
        adj_plot = adjustable_plot(chrome, fig, grid, idx, widgets);
        plots.append(adj_plot);
        
    plt.show()

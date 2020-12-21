# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:56:04 2020

@author: Paolo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import utilities as utl


class adjustable_plot():

    def __init__(self, chrome, fig, grid, idx, widgets):
        self.idx = idx;
        self.chrome = chrome
        self.fig = fig;
        self.freq = 0;
        self.ax = plt.subplot(grid[widgets*idx:widgets*(idx+1):, 0]);
        self.ax.margins(x=0)
        
        freqs = chrome.n_freqs();
        timepoints = chrome.n_time();
        
        default_start = 0;
        default_end = timepoints;
        
        y = get_freq_slice(chrome,self.freq,default_start, default_end);
        x = range(timepoints);
        self.ylim = np.max(chrome.data);
        
        self.plot, = self.ax.plot(x, y)
        self.ax.set_ylim(0, self.ylim)
        
        
        self.lines,  = self.ax.plot((0,100), (self.ylim, self.ylim));
        
        #set sliders
        axstart = plt.subplot(grid[widgets*idx + 1, 1])
        axend = plt.subplot(grid[widgets*idx + 2, 1])
        resetax = plt.subplot(grid[widgets*idx + 0, 1])
        
        
        axpeak_start = plt.subplot(grid[widgets*idx + 1, 2])
        axpeak_end = plt.subplot(grid[widgets*idx + 2, 2])


        #set sliders
        self.start_slider = Slider(axstart, 'start', 0, len(x), valinit=default_start)
        self.end_slider = Slider(axend, 'end', 0, timepoints , valinit=default_end)
        self.button = Button(resetax, 'Reset', hovercolor='0.975')
        
        #peak sliders
        self.peak_start_slider = Slider(axpeak_start, "peak-left", 0, len(x), valinit=default_start);
        self.peak_end_slider = Slider(axpeak_end, "peak-rigth", 0, timepoints, valinit=default_end);

        self.peak_start_slider.on_changed(self.update_peaks)
        self.peak_end_slider.on_changed(self.update_peaks)
        
        self.start_slider.on_changed(self.update)
        self.end_slider.on_changed(self.update)
        self.button.on_clicked(self.reset)

    def update(self, val):
        start = int(self.start_slider.val)
        end = int(self.end_slider.val)
        self.peak_start_slider.valmin = start;
        self.peak_start_slider.valmax = end;
        self.peak_end_slider.valmin = start;
        self.peak_end_slider.valmax = end;
        self.update_peaks(0);
        
        self.chrome.set_start_idx(start);
        self.chrome.set_end_idx(end);

        
        new_y = get_freq_slice(self.chrome, self.freq, start, end);
        new_x = range(end - start);
        self.plot.set_data(new_x, new_y)
        self.redraw();
        
    def redraw(self):  
        self.ax.relim()
        self.ax.autoscale_view(True,True,True)
        self.fig.canvas.draw_idle()     
    
    
    def update_peaks(self,val):
        self.lines.set_xdata([self.peak_start_slider.val, self.peak_start_slider.val]);
        self.redraw();
    
    def update_freq(self,val):
        if (int(val) > 0) and (int(val) < self.chrome.n_freqs()):
            self.freq = int(val);
            self.update(0);
        else: 
            print("WARNING: invalid frequency");
    
    def reset(self, event):
            self.start_slider.reset();
            self.end_slider.reset();
            self.freq = 0;

def get_freq_slice(chrome, freq, start, end):
    return chrome.data[start:end, freq];

def experiment_UI(exp):
    fig = plt.figure();
    
    n_chromes = exp.n_chromes();
    widgets = 4;
    grid = plt.GridSpec(n_chromes*widgets + 1, 3, wspace = .25, hspace = .20)

    axfreq = plt.subplot(grid[n_chromes*widgets,1])
    freq_box = TextBox(axfreq, 'Freq', initial = "0");
    

    plots = []
    for idx, chrome in enumerate(exp.chromes):
        adj_plot = adjustable_plot(chrome, fig, grid, idx, widgets);
        freq_box.on_submit(adj_plot.update_freq)
        plots.append(adj_plot);
        
    plt.show()

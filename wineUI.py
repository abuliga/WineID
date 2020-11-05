# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:56:04 2020

@author: Paolo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import utilities as utl


def get_freq_slice(chrome, freq, start, end):
    return chrome.data[start:end, freq];


def show_chromatogram(chrome):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    timepoints = chrome.n_time();
    freqs = chrome.n_freqs();

    def_freq = 0;
    def_start_time = 0;
    def_end_time = timepoints;
    
    y = get_freq_slice(chrome,def_freq,def_start_time, def_end_time);
    times = range(timepoints);
    
    l, = plt.plot(times, y)
    ax.set_ylim(0, np.max(chrome.data))
    ax.margins(x=0)
    
    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axstart = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axend = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

    
    s_freq = Slider(axfreq, 'Freq', 0, freqs, valinit=def_freq)
    s_start_time = Slider(axstart, 'start', 0, timepoints, valinit=def_start_time)
    s_end_time = Slider(axend, 'end', 0, timepoints , valinit=def_end_time)
    
    def update(val):
        start = int(s_start_time.val)
        end = int(s_end_time.val)
        freq = int(s_freq.val)
        new_y = get_freq_slice(chrome, freq, start, end);
        new_x = range(end - start);
        l.set_data(new_x, new_y)
        ax.relim()
        ax.autoscale_view(True,True,True)
        fig.canvas.draw_idle()




        
    s_freq.on_changed(update)
    s_start_time.on_changed(update)
    s_end_time.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    
    def reset(event):
        s_freq.reset()
        s_start_time.reset()
        
    button.on_clicked(reset)
    
    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
    
    
    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    radio.on_clicked(colorfunc)
    
    # Initialize plot with correct initial active value
    colorfunc(radio.value_selected)
    
    plt.show()
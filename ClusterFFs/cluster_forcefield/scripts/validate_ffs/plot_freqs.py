#! /usr/bin/env python3

from __future__ import print_function

import os

from optparse import OptionParser
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def compare(freq_ai, freq_ff, fn_fig):
    freqs_rmsd = np.sqrt(((freq_ff - freq_ai)**2).mean())
    freqs_md   = (freq_ff - freq_ai).mean()

    # Figure
    fig, ax = plt.subplots()

    cmap = ax.hexbin(freq_ai, freq_ff, cmap = 'inferno', mincnt = 1, norm = colors.LogNorm(), gridsize = 250)
    ax.plot([-100,3500],[-100,3500], 'k--', linewidth = 1.0) # Create reference line x = y
    # Set limits
    ax.set_xlim(-100,3500)
    ax.set_ylim(-100,3500)
    # Set labels
    ax.set_xlabel("Ab initio frequencies [cm$^{-1}$]")
    ax.set_ylabel("NMA frequencies [cm$^{-1}$]")

    ## Create zoomed inset plot
    zoom = 2.8
    axins = zoomed_inset_axes(ax, zoom, loc = 'upper left', bbox_to_anchor = (0.05, 1.0), bbox_transform = ax.transAxes) 
    axins.yaxis.tick_right()
    cmap_zoom = axins.hexbin(freq_ai, freq_ff, cmap = 'inferno', mincnt = 1, norm = colors.LogNorm(), gridsize = 250)
    axins.plot([-100,3500],[-100,3500], 'k--', linewidth = 1.0) # Create reference line x = y
    # Set labels
    axins.set_xlim(0, 500)
    axins.set_ylim(0, 500)

    # Draw where the inset comes from
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Write RMSD, MD and RMSE
    text = '\n'.join(['RMSD = {: 5.3e}'.format(freqs_rmsd), 'MD = {: 5.3e}'.format(freqs_md)])#, 'RMSE = {: 5.3e}'.format(freqs_rmse)])
    props = dict(boxstyle='round', facecolor = 'none', alpha=0.5)
    ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize=12, va = 'bottom', ha = 'right', bbox=props)
    plt.colorbar(cmap, ax = ax)
    plt.savefig(fn_fig, bbox_inches = 'tight')
    plt.close(fig)


def read_file(fn):
    freqs = []
    with open(fn, 'r') as f:
        for line in f.readlines()[1:]:
            freqs.append(float(line.strip()))
    return freqs

def iter_todo():
    for sbu in sorted(os.listdir('../../data')):
        if sbu.endswith('_extended'): continue
        result = []
        for key in ['qff', 'uff', 'ai']:
            fn_out = '../../data/{}/validation/frequencies/{}_freqs_{}.txt'.format(sbu, sbu, key)
            result.append(fn_out)
        yield result

if __name__ == '__main__':
    freqs = {key: [] for key in ['qff', 'uff', 'ai']}
    for fn_qff, fn_uff, fn_ai in iter_todo():
        freqs_qff = read_file(fn_qff)
        freqs_uff = read_file(fn_uff)
        freqs_ai = read_file(fn_ai)
        freqs['qff'].extend(freqs_qff)
        freqs['uff'].extend(freqs_uff)
        freqs['ai'].extend(freqs_ai)
    for key in ['qff', 'uff']:
        freqs_ff = np.array(freqs[key])
        freqs_ai = np.array(freqs['ai'])
        compare(freqs_ff, freqs_ai, '../../figs/validate_ffs/{}_freqs.pdf'.format(key))


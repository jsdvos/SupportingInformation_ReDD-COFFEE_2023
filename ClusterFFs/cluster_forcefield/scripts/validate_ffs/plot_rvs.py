#! /usr/bin/env python3

from __future__ import print_function

import os

from optparse import OptionParser
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from molmod.units import angstrom, deg
from molmod.ic import _bond_length_low, _bend_angle_low, _dihed_angle_low, _opdist_low

from yaff.system import System
from yaff import log
log.set_level(0)

def compare(ic_ff, ic_ai, figs_compare):
    rest_value_names = ['BONDS', 'BENDS', 'DIHEDRALS', 'OOPDISTS']
    unit_names = ['$\mathrm{\AA}$', '$^{\circ}$', '$^{\circ}$', '$\mathrm{\AA}$']
    units = [angstrom, deg, deg, angstrom]
    for i in range(4):
        rest_value_name = rest_value_names[i]
        unit = units[i]
        unit_name = unit_names[i]
        rest_value_ff = np.array(ic_ff[i])/unit
        rest_value_ai = np.array(ic_ai[i])/unit
        
        # Plot rest values
        if rest_value_name in ['BONDS', 'BENDS', 'DIHEDRALS', 'OOPDISTS'] and len(rest_value_ff) > 0:
            rmsd = np.sqrt(((rest_value_ff - rest_value_ai)**2).mean())
            md   = (rest_value_ff - rest_value_ai).mean()
            props = dict(boxstyle='round', facecolor = 'none', alpha=0.5)
            text = '\n'.join(['RMSD = {: 5.3e}'.format(rmsd), 'MD = {: 5.3e}'.format(md)])
            fig, ax = plt.subplots()
            ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize=12, va = 'bottom', ha = 'right', bbox=props)
            cmap = ax.hexbin(rest_value_ai, rest_value_ff, cmap = 'inferno', mincnt = 1, norm = colors.LogNorm(), gridsize = 250)
            lims = plt.xlim()
            ax.plot(lims, lims, 'k--', linewidth = 1.0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(r'Ab Initio rest {} [{}]'.format(rest_value_name.lower(), unit_name))
            ax.set_ylabel(r'Force Field rest {} [{}]'.format(rest_value_name.lower(), unit_name))
            fig.colorbar(cmap, ax = ax)
            fig.savefig(figs_compare[i], bbox_inches = 'tight')
            plt.close(fig)

def correct_dihedrals(diheds_0, diheds_1):
    corrected_diheds_0 = []
    corrected_diheds_1 = []
    for dihed_0, dihed_1 in zip(diheds_0, diheds_1):
        if abs(dihed_0 - dihed_1) > 180*deg:
            if dihed_0 > 0 and dihed_1 < 0:
                dihed_1 += 360*deg
            elif dihed_0 < 0 and dihed_1 > 0:
                dihed_0 += 360*deg
            else:
                assert False
        corrected_diheds_0.append(dihed_0)
        corrected_diheds_1.append(dihed_1)
    return corrected_diheds_0, corrected_diheds_1

def read_file(fn):
    ffatypes, ics = [], []
    with open(fn, 'r') as f:
        for line in f.readlines()[1:]:
            ffatype, ic = line.strip().split()
            ic = float(ic)
            ffatypes.append(ffatype)
            ics.append(ic)
    return ffatypes, np.array(ics)

def iter_todo():
    for sbu in sorted(os.listdir('../../data')):
        if sbu.endswith('_extended'): continue
        result = []
        for key in ['qff', 'uff', 'ai']:
            fns_out = ['../../data/{}/validation/rest_values/{}_rvs_{}_{}.txt'.format(sbu, sbu, key, ic) for ic in ['bond', 'bend', 'dihed', 'oop']]
            result.append(fns_out)
        yield result

if __name__ == '__main__':
    units = [angstrom, deg, deg, angstrom]
    ics = {key: ([], [], [], []) for key in ['qff', 'uff', 'ai']}
    for fns_qff, fns_uff, fns_ai in iter_todo():
        for i in range(4):
            ffatypes_qff, ics_qff = read_file(fns_qff[i])
            ffatypes_uff, ics_uff = read_file(fns_uff[i])
            ffatypes_ai, ics_ai = read_file(fns_ai[i])
            assert ffatypes_qff == ffatypes_uff == ffatypes_ai
            ics['qff'][i].extend(ics_qff*units[i])
            ics['uff'][i].extend(ics_uff*units[i])
            ics['ai'][i].extend(ics_ai*units[i])
    for key in ['qff', 'uff']:
        ics_ff = deepcopy(ics[key])
        ics_ai = deepcopy(ics['ai'])
        dihed_ff, dihed_ai = correct_dihedrals(ics_ff[2], ics_ai[2])
        ics_ff = (ics_ff[0], ics_ff[1], dihed_ff, ics_ff[3])
        ics_ai = (ics_ai[0], ics_ai[1], dihed_ai, ics_ai[3])
        fig_names = ['../../figs/validate_ffs/{}_{}.pdf'.format(key, ic) for ic in ['bond', 'bend', 'dihed', 'oop']]
        compare(ics_ff, ics_ai, fig_names)


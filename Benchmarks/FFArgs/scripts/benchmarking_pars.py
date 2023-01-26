import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import h5py

from yaff import System, log
from yaff.pes.ff import ForceField
from yaff.sampling.dof import FullCellDOF
from yaff.sampling.opt import CGOptimizer
from molmod.units import angstrom, kjmol

src_path = '../data'

rcut_default = 11.0*angstrom
gcut_default = 0.26/angstrom
alpha_default = 0.26/angstrom

for fn_chk in os.listdir(src_path):
    if not fn_chk.endswith('.chk'): continue
    struct = fn_chk.split('_optimized.chk')[0]
    fn_chk = os.path.join(src_path, fn_chk)
    fn_pars = os.path.join(src_path, '{}_pars.txt'.format(struct))
    f = h5py.File(fn_chk.replace('.chk', '.h5'), 'w')
    
    sys = System.from_file(fn_chk)

    # rcut
    energies = []
    rcut_range = np.arange(5, 20)*angstrom
    for rcut in rcut_range:
        rcut = rcut
        alpha_scale = alpha_default*rcut
        gcut_scale = gcut_default/alpha_default
        ff = ForceField.generate(sys, fn_pars, rcut = rcut, gcut_scale = gcut_scale, alpha_scale = alpha_scale, tailcorrections = True, smooth_ei = True)
        e = ff.compute()
        energies.append(e)
    results_rcut = f.create_group(struct + '/rcut')
    results_rcut['x_range'] = np.array(rcut_range)
    results_rcut['energies'] = np.array(energies)

    # gcut
    energies = []
    gcut_range = np.arange(0.2, 0.32, 0.01)/angstrom
    for gcut in gcut_range:
        rcut = rcut_default
        alpha_scale = alpha_default*rcut_default
        gcut_scale = gcut/alpha_default
        ff = ForceField.generate(sys, fn_pars, rcut = rcut, gcut_scale = gcut_scale, alpha_scale = alpha_scale, tailcorrections = True, smooth_ei = True)
        e = ff.compute()
        energies.append(e)
    results_gcut = f.create_group(struct + '/gcut')
    results_gcut['x_range'] = np.array(gcut_range)
    results_gcut['energies'] = np.array(energies)

    # alpha
    energies = []
    alpha_range = np.arange(0.1, 0.7, 0.04)/angstrom
    for alpha in alpha_range:
        rcut = rcut_default
        alpha_scale = alpha*rcut_default
        gcut_scale = gcut_default/alpha
        ff = ForceField.generate(sys, fn_pars, rcut = rcut, gcut_scale = gcut_scale, alpha_scale = alpha_scale, tailcorrections = True, smooth_ei = True)
        e = ff.compute()
        energies.append(e)
    results_alpha = f.create_group(struct + '/alpha')
    results_alpha['x_range'] = np.array(alpha_range)
    results_alpha['energies'] = np.array(energies)

    # plot
    f = h5py.File(fn_chk.replace('.chk', '.h5'), 'r')
    for key in ['rcut', 'gcut', 'alpha']:
        results = f[struct + '/' + key]
        if key == 'rcut':
            x_unit = angstrom
            xlabel = r'$r_\mathregular{cut}$ [$\mathregular{\AA}$]'
            default = rcut_default
            inset = True
            inset_start = 5
            inset_stop = -1
            xins_min = 10*angstrom
            xins_max = 20*angstrom
            x0 = 0.55
            y0 = 0.55
            width = 0.42
            height = 0.42
        if key == 'gcut':
            x_unit = 1/angstrom
            xlabel = r'$g_\mathregular{cut}$ [$\mathregular{1/\AA}$]'
            default = gcut_default
            inset = True
            inset_start = 5
            inset_stop = -1
            xins_min = 0.25/angstrom
            xins_max = 0.32/angstrom
            x0 = 0.7
            y0 = 0.1
            width = 0.27
            height = 0.27
        if key == 'alpha':
            x_unit = 1/angstrom
            xlabel = r'$\alpha$ [$\mathregular{1/\AA}$]'
            default = alpha_default
            inset = True
            inset_start = 3
            inset_stop = 6
            xins_min = 0.22/angstrom
            xins_max = 0.30/angstrom
            x0 = 0.45
            y0 = 0.1
            width = 0.3
            height = 0.3
        x = results['x_range'][:]
        e = results['energies'][:]
    
        # Plot energy
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Energy [kJ/mol]')
        ax.plot(x/x_unit, (e-e[-1])/kjmol, marker = 'o')
        ymin, ymax = ax.get_ylim()
        ax.vlines(default/x_unit, ymin, ymax, linestyle = '--')
        ax.set_ylim(ymin, ymax)
        if inset:
            axins = ax.inset_axes([x0, y0, width, height])
            axins.plot(x[inset_start:inset_stop]/x_unit, (e[inset_start:inset_stop]-e[-1])/kjmol, marker = 'o')
            axins.set_xlim(xins_min/x_unit, xins_max/x_unit)
            ymin, ymax = axins.get_ylim()
            axins.vlines(default/x_unit, ymin, ymax, linestyle = '--')
            axins.set_ylim(ymin, ymax)
        plt.savefig('../figs/{}_{}.pdf'.format(struct, key), bbox_inches = 'tight')
        plt.close()

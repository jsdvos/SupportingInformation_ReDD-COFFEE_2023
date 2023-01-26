import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from yaff import angstrom, kelvin, pascal, bar, boltzmann, kjmol

def get_fn_out(cof, pressure, host_guest, guest_guest):
    path_out = os.path.join('../../data/output_files', cof, '{}_{}_298K_{}Pa'.format(host_guest, guest_guest, pressure), 'Output/System_0/')
    fn_out = None
    for fn in os.listdir(path_out):
        if fn.endswith('.data'):
            fn_out = os.path.join(path_out, fn)
    return fn_out

def get_detail(cycles, ads, energies, stop):
    cycles = cycles[:stop]
    ads = ads[:stop]
    energies = energies[:stop]
    return cycles, ads, energies

def get_adsorption(cof, pressure, host_guest, guest_guest, unit = 'mol/uc'):
    fn = get_fn_out(cof, pressure, host_guest, guest_guest)
    temperature, pressure_, volume, supercell, cycles, abs_ads = read_raspa_output(fn, unit = unit)
    return abs_ads[50:100].mean()

def read_raspa_output(fn, unit = 'mol/uc'):
    with open(fn, 'r') as f:
        cycle_arr= []
        abs_ads_arr = []
        while True:
            line = f.readline()
            if not line: break
            # Temperature
            if line.startswith('External temperature:'):
                words = line.strip().split()
                assert words[3] == '[K]'
                temperature = float(words[2])*kelvin
            # Pressure
            if line.startswith('External Pressure:'):
                words = line.strip().split()
                assert words[3] == '[Pa]'
                pressure = float(words[2])*pascal
            # Simulation box dimensions
            if line.startswith('Number of unitcells ['):
                words = line.strip().split()
                if words[3] == '[a]:':
                    a = int(words[4])
                elif words[3] == '[b]:':
                    b = int(words[4])
                elif words[3] == '[c]:':
                    c = int(words[4])
            # Volume
            if line.startswith('volume of the cell:'):
                # Volume
                words = line.strip().split()
                assert words[5] == '(A^3)'
                volume = float(words[4])*angstrom**3
            # Cycles
            if line.startswith('Current cycle:'):
                # Per cycle: step, adsorbates, energy (in parts?)
                # abs_ads contains number of molecules per unit cell
                # nguests contains number of molecules per simulation box
                # there can be more unit cells in the simulation box, depending on 
                # the unit cell size and the cutoff radius
                # abs_ads can be converted to mg/g by multiplying it with a factor
                # mass(molecule)/mass(material)*1000
                # 
                # Start with current cycle
                words = line.strip().split()
                cycle = int(words[2])
                for i in range(13):
                    f.readline()
                words = f.readline().strip().split()
                assert ' '.join(words[:2]) == 'absolute adsorption:'
                assert words[3] == words[7] == words[11] == '(avg.'
                assert words[5] == '[mol/uc],'
                assert words[9] == '[mol/kg],'
                assert words[13] == '[mg/g]'
                if unit in ['mol/uc', 'mol/kg', 'mg/g']:
                    ads_index = {'mol/uc': 2, 'mol/kg': 6, 'mg/g': 10}[unit]
                    next_readlines = 6
                else:
                    words = f.readline().strip().split()
                    assert words[3] == words[8] == '[cm^3'
                    assert words[4] == 'STP/g],'
                    assert words[9] == 'STP/cm^3]'
                    ads_index = {'cm^3 STP/g': 0, 'cm^3 STP/cm^3': 5}[unit]
                    next_readlines = 5
                abs_ads = float(words[ads_index])
                for i in range(next_readlines):
                    f.readline()
                words = f.readline().strip().split()
                assert ' '.join(words[:3]) == 'Number of Adsorbates:'
                nguests = float(words[3])
                for i in range(2):
                    f.readline()
                words = f.readline().strip().split()
                assert ' '.join(words[:4]) == 'Current total potential energy:'
                cycle_arr.append(cycle)
                abs_ads_arr.append(abs_ads)
    supercell = (a, b, c)
    cycle_arr = np.array(cycle_arr)
    abs_ads_arr = np.array(abs_ads_arr)
    return temperature, pressure, volume, supercell, cycle_arr, abs_ads_arr

def plot_equilibration(cof, pressure, unit = 'mol/uc', detail = False):
    if detail:
        fn_fig = '../../figs/Equilibration_{}_{}Pa_det.pdf'.format(cof, pressure)
    else:
        fn_fig = '../../figs/Equilibration_{}_{}Pa.pdf'.format(cof, pressure)
    plt.figure()
    plt.xlabel('Cycle')
    plt.ylabel('Abs. adsorption [{}]'.format(unit))
    for host_guest in hgs:
        for guest_guest in ggs:
            fn_out = get_fn_out(cof, pressure, host_guest, guest_guest)
            temperature, pressure_, volume, supercell, cycles, abs_ads = read_raspa_output(fn_out, unit = unit)
            if detail:
                cycles, abs_ads, energies = get_detail(cycles, abs_ads, energies, 100)
            assert pressure == pressure_/pascal
            plt.plot(cycles, abs_ads, color = colors[guest_guest], linestyle = styles[host_guest])
    plt.savefig(fn_fig, bbox_inches = 'tight')
    plt.close()

def plot_isotherm(cof, unit = 'mg/g', fn_exp = None):
    fn_fig = '../../figs/Isotherm_{}.pdf'.format(cof)
    if fn_exp is not None and unit != 'mg/g':
        print('Asked for unit {}, but experimental curve is given in mg/g. Switching units to mg/g')
        unit = 'mg/g'
    plt.figure()
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('Adsorption [{}]'.format(unit))
    for hg in hgs:
        for gg in ggs:
            ads = []
            for pressure in pressures:
                ads.append(get_adsorption(cof, pressure, hg, gg, unit = unit))
            plt.plot(pressures, ads, color = colors[gg], linestyle = styles[hg])
    if fn_exp is not None:
        with open(fn_exp, 'r') as f:
            ads_arr = []
            pressure_arr = []
            for line in f:
                x, y = [float(x) for x in line.split(',')]
                pressure_arr.append(x*bar/pascal)
                ads_arr.append(y)
            plt.plot(pressure_arr, ads_arr, color = 'black', marker = '+')
    plt.savefig(fn_fig, bbox_inches = 'tight')
    plt.close()
            
if __name__ == '__main__':
    cofs = ['COF-1', 'COF-5', 'COF-102', 'COF-103']
    pressures = [0, 500000, 1000000, 1500000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000]
    hgs = ['mm3', 'uff', 'dreiding']

    # Define handles
    styles = {'mm3': '-', 'uff': '--', 'dreiding': ':'}
    ggs = ['mm3', 'uff', 'dreiding', 'trappe_ua', 'trappe_eh']
    colors = {'mm3': 'C0', 'uff': 'C1', 'dreiding': 'C2', 'trappe_ua': 'C3', 'trappe_eh': 'C4'}
    labels = {'mm3': 'MM3', 'uff': 'UFF', 'dreiding': 'DREIDING', 'trappe_ua': 'TraPPE-UA', 'trappe_eh': 'TraPPE-EH'}
    handles_gg = [mlines.Line2D([], [], color = colors[x], label = labels[x]) for x in ggs]
    handles_hg = [mlines.Line2D([], [], color = 'black', linestyle = styles[x], label = labels[x]) for x in hgs]
    handles = handles_gg + handles_hg
    
    # Create plots
    for cof in cofs:
        plot_isotherm(cof, fn_exp = '../../data/exp/{}_UptakeCH4.dat'.format(cof), label = False)
        for pressure in pressures:
            plot_equilibration(cof, pressure, detail = True, label = False)


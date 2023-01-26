import os

import numpy as np
import pandas as pd
import re

from molmod.units import angstrom, kelvin, pascal, bar, kjmol
from molmod.constants import avogadro, boltzmann

#press = 6500000
press = 580000

src_path = '../../data/298K_{}Pa'.format(press)

# STEP 1: iterate over structures
def iter_struct(fn = None, nstructs = None):
    for struct in os.listdir(src_path):
        yield struct

# STEP 2: get data filename
def get_fn_data(struct):
    output_path = os.path.join(src_path, struct, 'Output/System_0')
    if not os.path.exists(output_path): return None
    count = -1
    fn = None
    for count, fn in enumerate(os.listdir(output_path)):
        pass
    if count == 0:
        return os.path.join(output_path, fn)
    else:
        return None

# STEP 3: read data file
def read_raspa_output(fn):
    with open(fn, 'r') as f:
        runtime = -1
        cycle_arr= []
        abs_ads_arr = []
        energy_arr = []
        conversions = {}
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
            # Conversions
            if line.startswith('\tConversion factor'):
                key = re.search('mol(.*):', line).group(1)
                key = {
                        'ecules/unit cell -> mol/kg':   'mol/uc->mol/kg',
                        'ecules/unit cell -> mg/g':     'mol/uc->mg/g',
                        'ecules/unit cell -> cm^3 STP/gr': 'mol/uc->cm3STP/g',
                        'ecules/unit cell -> cm^3 STP/cm^3': 'mol/uc->cm3STP/cm3',
                        '/kg -> cm^3 STP/gr': 'mol/kg->cm3STP/g',
                        '/kg -> cm^3 STP/cm^3': 'mol/kg->cm3STP/cm3'
                        }[key]
                if key.startswith('mol/uc'):
                    conversions[key] = float(line.strip().split()[-2])
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
                abs_ads = float(words[2])
                for i in range(6):
                    f.readline()
                words = f.readline().strip().split()
                assert ' '.join(words[:3]) == 'Number of Adsorbates:'
                nguests = float(words[3])
                assert round(abs_ads*a*b*c, 3) == nguests, '{}*{}*{}*{} != {}'.format(abs_ads, a, b, c, nguests)
                for i in range(2):
                    f.readline()
                words = f.readline().strip().split()
                assert ' '.join(words[:4]) == 'Current total potential energy:'
                energy = float(words[4])*boltzmann
                cycle_arr.append(cycle)
                abs_ads_arr.append(abs_ads)
                energy_arr.append(energy)
            # Runtime
            if line.startswith('total time:'):
                words = line.strip().split()
                assert words[3] == '[s]'
                runtime = float(words[2])
    conversions['mol/uc->mol/sc'] = a*b*c
    cycle_arr = np.array(cycle_arr)
    abs_ads_arr = np.array(abs_ads_arr)
    energy_arr = np.array(energy_arr)
    if runtime == -1:
        runtime = -cycle_arr[-1] # Gives an indication of how fast the code ran, but is clearly off
    return temperature, pressure, volume, runtime, cycle_arr, abs_ads_arr, energy_arr, conversions

def read(fn_data):
    temp, pressure, vol, runtime, cycle_arr, abs_ads_arr, energy_arr, conversions = read_raspa_output(fn_data)
    assert temp == 298.*kelvin
    assert pressure == float(press)*pascal
    
    # Reduce array
    if len(cycle_arr) >= 100:
        abs_ads_arr = abs_ads_arr[50:100]
        energy_arr = energy_arr[50:100]
    else:
        return None, None

    # Convert to units
    abs_ads_dict = {}
    for unit in ['mol/uc', 'mol/sc', 'mg/g', 'mg/cm3', 'mol/kg', 'cm3STP/g', 'cm3STP/cm3']:
        if unit == 'mol/uc':
            abs_ads_dict[unit] = abs_ads_arr
        elif unit == 'mg/cm3':
            density = conversions['mol/uc->cm3STP/cm3']/conversions['mol/uc->cm3STP/g']
            abs_ads_dict[unit] = abs_ads_arr*conversions['mol/uc->mg/g']*density
        else:
            abs_ads_dict[unit] = abs_ads_arr*conversions['mol/uc->' + unit]

    return abs_ads_dict, energy_arr

def heat_of_adsorption(abs_ads, energy, temp):
    if min(abs_ads) == max(abs_ads) == 0:
        q = 0.0
    else:
        en = (abs_ads*energy).mean()
        n = abs_ads.mean()
        e = energy.mean()
        n2 = (abs_ads*abs_ads).mean()
        q = (en - e*n)/(n2 - n**2)
    q_st = - q + boltzmann*temp
    return q_st
    
if __name__ == '__main__':
    units = ['mol/uc', 'mol/sc', 'mg/g', 'mg/cm3', 'mol/kg', 'cm3STP/g', 'cm3STP/cm3']
    data = {'struct': [], 'q_st [kjmol]': []}
    data.update({'abs_ads [{}]'.format(unit): [] for unit in units})
    for struct in iter_struct():
        fn_data = get_fn_data(struct)
        if fn_data is None:
            # Wrong file
            continue
        abs_ads, energy = read(fn_data)
        if abs_ads is None and energy is None:
            # Corrupt file
            continue
        q_st = heat_of_adsorption(abs_ads['mol/sc'], energy, 298*kelvin)
        
        data['q_st [kjmol]'].append(q_st/kjmol)
        data['struct'].append(struct)
        for unit in units:
            data['abs_ads [{}]'.format(unit)].append(abs_ads[unit].mean())
    columns = ['struct'] + ['abs_ads [{}]'.format(unit) for unit in units] + ['q_st [kjmol]']
    df = pd.DataFrame(data, columns = columns)
    df.to_csv('../../data/CH4_298K_{}Pa.csv'.format(press), sep = ';')

        


    

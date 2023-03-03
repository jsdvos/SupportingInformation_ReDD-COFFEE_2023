import sys

import numpy as np
import h5py

from yaff import System
from molmod.units import angstrom, deg
from molmod.ic import _bond_length_low, _bend_angle_low, _dihed_angle_low, _opdist_low

def get_relative(system, i, j):
    dr_ij = system.pos[i] - system.pos[j]
    if system.cell.nvec > 0:
        system.cell.mic(dr_ij)
    return dr_ij

def get_bond_length(system, i, j):
    # Value
    bond = get_relative(system, i, j)
    value = _bond_length_low(bond, 0)[0]
    # FFatype
    itype = system.get_ffatype(i)
    jtype = system.get_ffatype(j)
    ffatype = '.'.join(sorted([itype, jtype]))
    return value, '1-' + ffatype

def get_bend_angle(system, i, j, k):
    # Value
    bond_ij = get_relative(system, i, j)
    bond_kj = get_relative(system, k, j)
    value = _bend_angle_low(bond_ij, bond_kj, 0)[0]
    # FFatype
    itype = system.get_ffatype(i)
    jtype = system.get_ffatype(j)
    ktype = system.get_ffatype(k)
    if itype < ktype:
        ffatype = '{}.{}.{}'.format(itype, jtype, ktype)
    else:
        ffatype = '{}.{}.{}'.format(ktype, jtype, itype)
    return value, '2-' + ffatype

def get_dihedral_angle(system, i, j, k, l):
    # Value
    bond_ij = get_relative(system, i, j)
    bond_kj = get_relative(system, k, j)
    bond_lk = get_relative(system, l, k)
    value = _dihed_angle_low(bond_ij, bond_kj, bond_lk, 0)[0]
    # FFatype
    itype = system.get_ffatype(i)
    jtype = system.get_ffatype(j)
    ktype = system.get_ffatype(k)
    ltype = system.get_ffatype(l)
    if itype < ltype:
        ffatype = '{}.{}.{}.{}'.format(itype, jtype, ktype, ltype)
    else:
        ffatype = '{}.{}.{}.{}'.format(ltype, ktype, jtype, itype)
    return value, '3-' + ffatype

def get_oop_dist(system, i, j, k, l):
    # l is the central atom
    # Value
    bond_il = get_relative(system, i, l)
    bond_jl = get_relative(system, j, l)
    bond_kl = get_relative(system, k, l)
    value = _opdist_low(bond_il, bond_jl, bond_kl, 0)[0]
    # FFatype
    itype = system.get_ffatype(i)
    jtype = system.get_ffatype(j)
    ktype = system.get_ffatype(k)
    ltype = system.get_ffatype(l)
    ffatype = '{}.{}'.format('.'.join(sorted([itype, jtype, ktype])), ltype)
    return value, '4-' + ffatype

def get_cell_params(system):
    if system.cell.nvec > 0:
        lengths, angles = system.cell.parameters
        volume = system.cell.volume
        return lengths, angles, volume
    else:
        return None, None, None

def iter_sys(cof, ff_label, method):
    if method == 'dyn':
        with h5py.File('../data/{}/{}_{}.h5'.format(cof, cof, ff_label), 'r') as f:
            # Static attributes
            numbers = f['system']['numbers'][:]
            bonds = f['system']['bonds'][:]
            ffatypes = f['system']['ffatypes'][:]
            ffatypes = np.array([ffatype.decode('UTF-8') for ffatype in ffatypes])
            ffatype_ids = f['system']['ffatype_ids'][:]
            assert (f['trajectory']['cell'][0] == f['system']['rvecs'][:]).all()
            # Dynamic attributes
            cells = f['trajectory']['cell'][100:]
            positions = f['trajectory']['pos'][100:]
        system = System(numbers = numbers, pos = positions[0], bonds = bonds, ffatypes = ffatypes, ffatype_ids = ffatype_ids, rvecs = cells[0])
        for i in range(len(cells)):
            system.pos[:] = positions[i]
            system.cell.update_rvecs(cells[i])
            yield system
    elif method == 'stat':
        yield System.from_file('../data/{}/{}_{}_opt.chk'.format(cof, cof, ff_label))
    elif method == 'exp':
        yield System.from_file('../data/{}/{}.chk'.format(cof, cof))

for cof in ['COF-300', 'LZU-111', 'COF-320_89K', 'COF-320_298K']:
    for ff_label in ['uff', 'qff']:
        for method in ['exp', 'stat', 'dyn']:
            result = {key: [] for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume']}
            for count, system in enumerate(iter_sys(cof, ff_label, method)):
                print(count)
                for i, j in system.iter_bonds():
                    bond_length, bond_ffatype = get_bond_length(system, i, j)
                    if count == 0:
                        if bond_ffatype not in result.keys():
                            result[bond_ffatype] = []
                    result[bond_ffatype].append(bond_length)
                for i, j, k in system.iter_angles():
                    bend_angle, bend_ffatype = get_bend_angle(system, i, j, k)
                    if count == 0:
                        if bend_ffatype not in result.keys():
                            result[bend_ffatype] = []
                    result[bend_ffatype].append(bend_angle)
                for i, j, k, l in system.iter_dihedrals():
                    dihed_angle, dihed_ffatype = get_dihedral_angle(system, i, j, k, l)
                    dihed_angle = abs(dihed_angle)
                    if dihed_angle > 90*deg:
                        dihed_angle = 180*deg - dihed_angle
                    if count == 0:
                        if dihed_ffatype not in result.keys():
                            result[dihed_ffatype] = []
                    result[dihed_ffatype].append(dihed_angle)
                for i, j, k, l in system.iter_oops():
                    oop_dist, oop_ffatype = get_oop_dist(system, i, j, k, l)
                    if count == 0:
                        if oop_ffatype not in result.keys():
                            result[oop_ffatype] = []
                    result[oop_ffatype].append(oop_dist)
                lengths, angles, volume = get_cell_params(system)
                result['a'].append(lengths[0])
                result['b'].append(lengths[0])
                result['c'].append(lengths[0])
                result['alpha'].append(angles[0])
                result['beta'].append(angles[0])
                result['gamma'].append(angles[0])
                result['volume'].append(volume)
            with open('../{}/{}_{}_{}.txt'.format(cof, cof, ff_label, method), 'w') as f:
                for key in sorted(result.keys()):
                    values = result[key]
                    f.write('{} {} {}\n'.format(key, np.mean(values), np.std(values)))

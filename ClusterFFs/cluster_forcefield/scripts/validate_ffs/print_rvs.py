#! /usr/bin/env python3

from __future__ import print_function

import os

from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from molmod.units import angstrom, deg
from molmod.ic import _bond_length_low, _bend_angle_low, _dihed_angle_low, _opdist_low

from yaff.system import System
from yaff import log
log.set_level(0)

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
    itype = '{}({})'.format(system.get_ffatype(i), i)
    jtype = '{}({})'.format(system.get_ffatype(j), j)
    ffatype = '.'.join(sorted([itype, jtype]))
    return value, ffatype

def get_bend_angle(system, i, j, k):
    # Value
    bond_ij = get_relative(system, i, j)
    bond_kj = get_relative(system, k, j)
    value = _bend_angle_low(bond_ij, bond_kj, 0)[0]
    # FFatype
    itype = '{}({})'.format(system.get_ffatype(i), i)
    jtype = '{}({})'.format(system.get_ffatype(j), j)
    ktype = '{}({})'.format(system.get_ffatype(k), k)
    if itype < ktype:
        ffatype = '{}.{}.{}'.format(itype, jtype, ktype)
    else:
        ffatype = '{}.{}.{}'.format(ktype, jtype, itype)
    return value, ffatype

def get_dihedral_angle(system, i, j, k, l):
    # Value
    bond_ij = get_relative(system, i, j)
    bond_kj = get_relative(system, k, j)
    bond_lk = get_relative(system, l, k)
    value = _dihed_angle_low(bond_ij, bond_kj, bond_lk, 0)[0]
    # FFatype
    itype = '{}({})'.format(system.get_ffatype(i), i)
    jtype = '{}({})'.format(system.get_ffatype(j), j)
    ktype = '{}({})'.format(system.get_ffatype(k), k)
    ltype = '{}({})'.format(system.get_ffatype(l), l)
    if itype < ltype:
        ffatype = '{}.{}.{}.{}'.format(itype, jtype, ktype, ltype)
    else:
        ffatype = '{}.{}.{}.{}'.format(ltype, ktype, jtype, itype)
    return value, ffatype

def get_oop_dist(system, i, j, k, l):
    # l is the central atom
    # Value
    bond_il = get_relative(system, i, l)
    bond_jl = get_relative(system, j, l)
    bond_kl = get_relative(system, k, l)
    value = _opdist_low(bond_il, bond_jl, bond_kl, 0)[0]
    # FFatype
    itype = '{}({})'.format(system.get_ffatype(i), i)
    jtype = '{}({})'.format(system.get_ffatype(j), j)
    ktype = '{}({})'.format(system.get_ffatype(k), k)
    ltype = '{}({})'.format(system.get_ffatype(l), l)
    ffatype = '{}.{}'.format('.'.join(sorted([itype, jtype, ktype])), ltype)
    return value, ffatype

def get_cell_params(system):
    if system.cell.nvec > 0:
        lengths, angles = system.cell.parameters
        volume = system.cell.volume
        return lengths, angles, volume
    else:
        return None, None, None

def get_rest_values(system, ffatypes = False):
    bonds = []
    bends = []
    diheds = []
    oops = []
    if ffatypes:
        bond_ffatypes = []
        bend_ffatypes = []
        dihed_ffatypes = []
        oop_ffatypes = []
    for i, j in system.iter_bonds():
        bond_length, bond_ffatype = get_bond_length(system, i, j)
        bonds.append(bond_length)
        if ffatypes:
            bond_ffatypes.append(bond_ffatype)
    for i, j, k in system.iter_angles():
        bend_angle, bend_ffatype = get_bend_angle(system, i, j, k)
        bends.append(bend_angle)
        if ffatypes:
            bend_ffatypes.append(bend_ffatype)
    for i, j, k, l in system.iter_dihedrals():
        dihed_angle, dihed_ffatype = get_dihedral_angle(system, i, j, k, l)
        diheds.append(dihed_angle)
        if ffatypes:
            dihed_ffatypes.append(dihed_ffatype)
    for i, j, k, l in system.iter_oops():
        oop_dist, oop_ffatype = get_oop_dist(system, i, j, k, l)
        oops.append(oop_dist)
        if ffatypes:
            oop_ffatypes.append(oop_ffatype)
    lengths, angles, volume = get_cell_params(system)
    if ffatypes:
        return bonds, bond_ffatypes, bends, bend_ffatypes, diheds, dihed_ffatypes, oops, oop_ffatypes, lengths, angles, volume
    else:
        return bonds, bends, diheds, oops, lengths, angles, volume

def dump_to_files(all_ffatypes, all_ics, fn_names, fn_sys, units = [angstrom, deg, deg, angstrom]):
    # Dump all internal coordinates to a file
    for i in range(4):
        ffatypes = all_ffatypes[i]
        ics = all_ics[i]
        fn_name = fn_names[i]
        unit = units[i]
        ic_types = ['Bond distances', 'Bend angles', 'Dihedral angles', 'Out-of-plane distances']
        unit_names = ['A', 'deg', 'deg', 'A']
        data = []
        for ffatype, ic in zip(ffatypes, ics):
            data.append([ffatype, ic/unit])
        data = np.array(data)
        np.savetxt(fn_names[i], data, fmt = '%s', header = '{} [{}] for the system {}'.format(ic_types[i], unit_names[i], fn_sys))

def iter_todo():
    for sbu in sorted(os.listdir('../../data')):
        for key in ['qff', 'uff', 'ai']:
            fn_chk = '../../data/{}/{}_opt_{}.chk'.format(sbu, sbu, key)
            fns_out = ['../../data/{}/validation/rest_values/{}_rvs_{}_{}.txt'.format(sbu, sbu, key, ic) for ic in ['bond', 'bend', 'dihed', 'oop']]
            yield fn_chk, fns_out

if __name__ == '__main__':
    for fn_chk, fns_out in iter_todo():
        system = System.from_file(fn_chk)
        bonds, bond_ffatypes, bends, bend_ffatypes, diheds, dihed_ffatypes, oops, oop_ffatypes, lengths, angles, volume = get_rest_values(system, ffatypes = True)
        ffatypes = [bond_ffatypes, bend_ffatypes, dihed_ffatypes, oop_ffatypes]
        ics = [bonds, bends, diheds, oops]
        dump_to_files(ffatypes, ics, fns_out, fn_chk)

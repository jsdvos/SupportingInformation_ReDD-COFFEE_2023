import os
from sys import argv

import numpy as np

from molmod.units import angstrom, meter, centimeter
from molmod.constants import lightspeed
from molmod.io.chk import load_chk

from yaff.system import System
from yaff.pes.ff import ForceField
from yaff.sampling.dof import CartesianDOF
from yaff.sampling.opt import CGOptimizer
from yaff.sampling.harmonic import estimate_cart_hessian
from yaff import log

from tamkin import Molecule, NMA, ConstrainExt
log.set_level(0)

def iter_sbus():
    if len(argv) == 1:
        raise IOError('Should specify which force field to use (QuickFF or UFF)')
    elif argv[1] in ['qff', 'QFF', 'quickff', 'QuickFF']:
        fns = ['pars_yaff.txt', 'pars_ei.txt', 'pars_mm3.txt']
        key = 'qff'
    elif argv[1] in ['uff', 'UFF']:
        fns = ['pars_cov_uff.txt', 'pars_ei.txt', 'pars_lj_uff.txt']
        key = 'uff'
    for sbu in sorted(os.listdir('../../data')):
        if sbu.endswith('_extended'): continue
        fns_pars = ['../../data/{}/{}'.format(sbu, fn) for fn in fns]
        fn_chk = '../../data/{}/validation/{}_opt_ai.chk'.format(sbu, sbu)
        fn_out = '../../data/{}/validation/{}_opt_{}.chk'.format(sbu, sbu, key)
        fn_freq = '../../data/{}/validation/frequencies/{}_freqs_{}.txt'.format(sbu, sbu, key)
        yield fn_chk, fns_pars, fn_out, fn_freq

def is_minimum(ff, fn_freq, threshold = 0.0):
    system = ff.system
    gpos = np.zeros(system.pos.shape, float)
    energy = ff.compute(gpos = gpos)
    hessian = estimate_cart_hessian(ff)
    molecule = Molecule(system.numbers, system.pos, system.masses, 0.0, gpos, hessian)
    try:
        nma = NMA(molecule, ConstrainExt(), do_modes = True)
        if nma.freqs[0] < -threshold:
            print(nma.freqs[0])
            pert = np.array([nma.modes[i][0] for i in range(len(nma.modes))])
            amplitude = 1*angstrom
            ff.update_pos(ff.system.pos + amplitude*pert.reshape(-1, 3))
            return False
        else:
            np.savetxt(fn_freq, nma.freqs/(lightspeed/centimeter), header = 'NMA Frequencies in cm-1')
            return True
    except ValueError:
        return False


if __name__ == '__main__':
    for fn_chk, fns_pars, fn_out, fn_freq in iter_sbus():
        sys = System.from_file(fn_chk)
        ff = ForceField.generate(sys, fns_pars, rcut = 1*meter, smooth_ei = True)
        while not is_minimum(ff, fn_freq):
            dof = CartesianDOF(ff)
            opt = CGOptimizer(dof)
            opt.run(10000)
        opt.dof.ff.system.to_file(fn_out)

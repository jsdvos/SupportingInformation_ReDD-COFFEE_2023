#! /usr/bin/env python2
import os

from optparse import OptionParser

from yaff.system import System
from quickff.io import read_abinitio
from quickff.reference import SecondOrderTaylor, YaffForceField
from quickff.program import DeriveFF
from quickff.settings import Settings
from quickff.log import log
from molmod.units import angstrom, meter
from yaff.pes.ff import ForceField


def parse():
    usage = '%prog [options] system.chk ai.fchk/ai.xml'
    descr = 'Create valence force field using QuickFF'
    parser = OptionParser(usage = usage, description = descr)
    parser.add_option(
            '-c', '--config', default = None,
            help = 'Config file with all QuickFF settings'
    )
    parser.add_option(
            '-p', '--plot_traj', action = 'store_true', default = False,
            help = 'Flag if the trajectories are plotted'
    )
    parser.add_option(
            '-e', '--ff_ei', default = None,
            help = 'Reference electrostatic FF'
    )
    parser.add_option(
            '-m', '--ff_mm3', default = None,
            help = 'Reference MM3 FF'
    )
    parser.add_option(
            '-o', '--output', default = None,
            help = 'Output valence force field name'
    )
    options, args = parser.parse_args()
    if not len(args) == 2 or not args[0].endswith('.chk') or not (os.path.splitext(args[1])[-1] in ['.fchk', '.xml']):
        raise IOError('Exactly two arguments expected: the CHK System file and the AI input (either Gaussian FCHK or VASP XML file)')
    fn_sys, fn_ai = args
    fn_config = options.config # If None, quickffrc is used
    fn_refs = [fn_ref for fn_ref in [options.ff_ei, options.ff_mm3] if not fn_ref == None]
    if options.output == None:
        path = os.path.dirname(fn_sys)
        fn_out = os.path.join(path, 'pars_yaff.txt')
    else:
        fn_out = options.output
    return fn_sys, fn_ai, fn_config, fn_refs, fn_out, options.plot_traj
    
def get_settings(fn_config, fn_out, plot_traj):
    if plot_traj:
        settings = Settings(fn = fn_config, fn_yaff = fn_out, fn_traj = 'trajectories.pp', plot_traj = 'final')
        log.section_level = -1
    else:
        settings = Settings(fn = fn_config, fn_yaff = fn_out)
    return settings

def get_system(fn_sys):
    system = System.from_file(fn_sys)
    system.detect_bonds()
    return system

def get_ai(fn_ai):
    numbers, coords, energy, grad, hess, masses, rvecs, pbc = read_abinitio(fn_ai)
    ai = SecondOrderTaylor('ai', coords = coords, energy = 0.0, grad = grad, hess = hess, pbc = pbc)
    return ai

def get_ff_refs(system, fn_refs):
    # Periodic system: rcut = 15*angstrom
    # Non-periodic system: rcut = 1*meter -> include all interactions
    if system.cell.nvec == 0:
        rcut = 1*meter
    else:
        rcut = 15*angstrom
    ff_refs = []
    for fn_ref in fn_refs:
        if 'ei' in fn_ref:
            ff_ei = ForceField.generate(system, fn_ref, rcut = rcut, smooth_ei = True)
            ei = YaffForceField('EI', ff_ei)
            ff_refs.append(ei)
        elif 'mm3' in fn_ref:
            ff_mm3 = ForceField.generate(system, fn_ref, rcut = rcut, alpha_scale = 3.2, gcut_scale = 1.5)
            mm3 = YaffForceField('MM3', ff_mm3)
            ff_refs.append(mm3)
        else:
            raise IOError('FF file {} not recognized'.format(fn_ref))
    return ff_refs

def main():
    fn_sys, fn_ai, fn_config, fn_refs, fn_out, plot_traj = parse()
    settings = get_settings(fn_config, fn_out, plot_traj)
    system = get_system(fn_sys)
    ai = get_ai(fn_ai)
    ffrefs = get_ff_refs(system, fn_refs)
    
    program = DeriveFF(system, ai, settings, ffrefs = ffrefs)
    program.run()

if __name__ == '__main__':
    main()

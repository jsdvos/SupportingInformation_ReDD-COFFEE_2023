import os
from sys import argv

import numpy as np

from molmod.periodic import periodic
from molmod.units import angstrom, deg, amu, kelvin, atm, pascal, kjmol, kcalmol
from molmod.constants import boltzmann
from yaff import System, log
log.set_level(0)
from yaff_ff import ForceField

def parse():
    args = argv[1:]
    assert len(args) == 7, 'Required 7 input parameters: 5 filenames + temperature [K] + pressure [Pa]'
    fn_host, fn_guest, fn_host_host, fn_host_guest, fn_guest_guest, temp, press = args
    temp = float(temp)*kelvin
    press = float(press)*pascal
    return fn_host, fn_guest, fn_host_host, fn_host_guest, fn_guest_guest, temp, press

def get_pars_from_ff(ff):
    for part in ff.parts:
        if not part.name in ['pair_mm3', 'pair_lj', 'pair_ljcross']: continue
        ff_type = ff_types[part.name]
        sigmas = {}
        epsilons = {}
        if ff_type == 'UFF':
            sig_cross = part.pair_pot.sig_cross.copy()
            eps_cross = part.pair_pot.eps_cross.copy()
            sigmas_ = np.diagonal(sig_cross)
            epsilons_ = np.diagonal(eps_cross)
            # Check that LJCross indeed represents UFF (LJ with Jorgensen mixing rules)
            nffatypes = len(ff.system.ffatypes)
            sig_cross_check = np.zeros([nffatypes, nffatypes])
            eps_cross_check = np.zeros([nffatypes, nffatypes])
            for i in range(nffatypes):
                for j in range(nffatypes):
                    sig_cross_check[i, j] = np.sqrt(sigmas_[i]*sigmas_[j])
                    eps_cross_check[i, j] = np.sqrt(epsilons_[i]*epsilons_[j])
            delta_sig = np.max(abs(sig_cross_check - sig_cross))
            delta_eps = np.max(abs(eps_cross_check - eps_cross))
            if delta_sig > 1e-6*angstrom or delta_eps > 1e-6*kcalmol:
                print('Assumed Jorgensen mixing rules for UFF, but these do not reproduce data')
            # Assign to ffatypes
            for i in range(nffatypes):
                sigmas[ff.system.ffatypes[i]] = sigmas_[i]
                epsilons[ff.system.ffatypes[i]] = epsilons_[i]
        else:
            sigmas_ = part.pair_pot.sigmas.copy()
            epsilons_ = part.pair_pot.epsilons.copy()
            for i in range(ff.system.natom):
                sigmas[ff.system.get_ffatype(i)] = sigmas_[i]
                epsilons[ff.system.get_ffatype(i)] = epsilons_[i]
        return sigmas, epsilons, ff_type

ff_types = {
        'pair_mm3': 'MM3',
        'pair_lj': 'DREIDING', # Can also be TraPPE, but same functional form and mixing rules
        'pair_ljcross': 'UFF'
        }

def read_host_guest(ff_host_host, ff_host_guest):
    # Get host-guest interactions
    ff_type = 'None'
    sigmas = {}
    epsilons = {}
    for ff in [ff_host_guest, ff_host_host]:
        sigmas_, epsilons_, ff_type_ = get_pars_from_ff(ff)
        assert ff_type in [ff_type_, 'None'], 'VdW potentials in {} and {} cannot be mixed'.format(fn_host_host, fn_host_guest)
        ff_type = ff_type_
        sigmas.update(sigmas_)
        epsilons.update(epsilons_)
    return sigmas, epsilons, ff_type

def read_guest_guest(ff_guest_guest):
    # Get guest-guest interactions
    sigmas_, epsilons_, ff_type = get_pars_from_ff(ff_guest_guest)
    sigmas = {}
    epsilons = {}
    for i in range(len(ff_guest_guest.system.ffatypes)):
        for j in range(i, len(ff_guest_guest.system.ffatypes)):
            ffatype_i = ff_guest_guest.system.ffatypes[i]
            ffatype_j = ff_guest_guest.system.ffatypes[j]
            if ff_type == 'UFF':
                # Jorgensen mixing rules for UFF
                sig = np.sqrt(sigmas_[ffatype_i]*sigmas_[ffatype_j])
                eps = np.sqrt(epsilons_[ffatype_i]*epsilons_[ffatype_j])
            else:
                # Lorentz-Berthelot mixing rules for DREIDING, MM3 and TraPPE
                sig = (sigmas_[ffatype_i] + sigmas_[ffatype_j])/2
                eps = np.sqrt(epsilons_[ffatype_i]*epsilons_[ffatype_j])
            sigmas[(ffatype_i, ffatype_j)] = sig
            epsilons[(ffatype_i, ffatype_j)] = eps
    return sigmas, epsilons, ff_type

def write_cif(host, workdir = '.', fn_out = 'framework.cif'):
    with open(os.path.join(workdir, fn_out), 'w') as f:
        f.write('data_{}\n'.format(fn_out.split('.')[0]))
        f.write('_symmetry_space_group_name_H-M       \'P1\'\n')
        f.write('_audit_creation_method            \'HORTON\'\n')
        f.write('_symmetry_Int_Tables_number       1\n')
        f.write('_symmetry_cell_setting            triclinic\n\n')
        f.write('loop_\n')
        f.write('_symmetry_equiv_pos_as_xyz\n')
        f.write('  x,y,z\n\n')
        lengths, angles = host.cell.parameters
        f.write('_cell_length_a     %12.6f\n' % (lengths[0]/angstrom))
        f.write('_cell_length_b     %12.6f\n' % (lengths[1]/angstrom))
        f.write('_cell_length_c     %12.6f\n' % (lengths[2]/angstrom))
        f.write('_cell_angle_alpha  %12.6f\n' % (angles[0]/deg))
        f.write('_cell_angle_beta   %12.6f\n' % (angles[1]/deg))
        f.write('_cell_angle_gamma  %12.6f\n\n' % (angles[2]/deg))
        f.write('loop_\n')
        f.write('_atom_site_label\n')
        f.write('_atom_site_type_symbol\n')
        f.write('_atom_site_fract_x\n')
        f.write('_atom_site_fract_y\n')
        f.write('_atom_site_fract_z\n')
        f.write('_atom_site_charge\n')
        for i in range(host.natom):
            gvecs_full = host.cell._get_gvecs(full=True)
            fx, fy, fz = np.dot(host.pos[i], gvecs_full.T)
            f.write('{:21s} {:3s} {: 12.6f} {: 12.6f} {: 12.6f} {: 12.6f}\n'.format(host.get_ffatype(i), periodic[host.numbers[i]].symbol, fx, fy, fz, host.charges[i]))

def write_pseudo_atoms(ff_host_host, ff_host_guest, workdir = '.'):
    all_lines = []
    for system in [ff_host_host.system, ff_host_guest.system]:
        nffatypes = len(system.ffatypes)
        lines = ['']*nffatypes
        for i in range(system.natom):
            ffatype = system.get_ffatype(i)
            number = system.numbers[i]
            if number == 0:
                symbol = 'M'
                mass = 0.0
            else:
                symbol = periodic[number].symbol
                mass = periodic[number].mass
            charge = system.charges[i]
            lines[system.ffatype_ids[i]] = "{:21s} {:6s} {:6s} {:6s} {:9d} {:12.8f} {:12.8f} {:12.3f}   {:9.3f} {:6.3f}  {:12d} {:12d} {:>16s} {:12d}\n".format(ffatype, 'yes', symbol, symbol, 0, mass/amu, charge,0.0,1.0,1.0,0,0,'absolute',0)
            if not '' in lines:
                break
        all_lines.extend(lines)
    with open(os.path.join(workdir, 'pseudo_atoms.def'), 'w') as f:
        f.write('# number of pseudo atoms\n')
        f.write('{}\n'.format(len(all_lines)))
        f.write("# {:19s} {:6s} {:6s} {:6s} {:9s} {:12s} {:12s} {:12s}    {:9s} {:6s} {:12s}  {:12s} {:16s} {:12s}\n".format('type', 'print', 'as', 'chem', 'oxidation', 'mass', 'charge', 'polarization', 'B-factor', 'radii', 'connectivity', 'anisotropic', 'anisotropic-type', 'tinker-type'))
        for line in all_lines:
            f.write(line)

def convert(sigma, epsilon, ff_type):
    if ff_type == 'MM3':
        potential = 'MM3_VDW'
        sigma = sigma/angstrom
        epsilon = epsilon/kcalmol
    elif ff_type == 'DREIDING':
        potential = 'LENNARD-JONES'
        sigma = sigma/angstrom
        epsilon = epsilon/boltzmann/kelvin
    elif ff_type == 'UFF':
        potential = 'LENNARD-JONES'
        sigma = sigma/angstrom
        epsilon = epsilon/boltzmann/kelvin
    return potential, sigma, epsilon

mixing_rules = {
        'UFF': 'Jorgensen',
        'DREIDING': 'Lorentz-Berthelot',
        'MM3': 'Lorentz-Berthelot'
        }

def write_force_field_mixing_rules(sigmas, epsilons, ff_type, workdir = '.'):
    # Host-guest interactions are written to force_field_mixing_rules.def
    with open(os.path.join(workdir, 'force_field_mixing_rules.def'), 'w') as f:
        f.write('# general rule for shifted vs truncated\n')
        f.write('truncated\n')
        f.write('# general rule tailcorrections\n')
        f.write('yes\n')
        f.write("# number of defined interactions\n")
        f.write('{}\n'.format(len(sigmas)))
        f.write('# {:19s}    {:15s}    {:6s}  {:6s}\n'.format('type', 'interaction', 'epsilon', 'sigma'))
        for ffatype in sorted(sigmas.keys(), key=lambda e: (e.split('_')[-1], e)):
            sigma = sigmas[ffatype]
            epsilon = epsilons[ffatype]
            if epsilon == 0.0:
                f.write('{:21s}    {:15s}\n'.format(ffatype, 'NONE'))
            else:
                potential, sigma, epsilon = convert(sigma, epsilon, ff_type)
                f.write('{:21s}    {:15s}    {:6.3f}  {:6.3f}\n'.format(ffatype, potential, epsilon, sigma))
        f.write("# Mixing rules (Lorentz-Berthelot: s = (s0+s1)/2; Jorgensen: s = sqrt(s0*s1); eps = sqrt(eps0*eps1))\n")
        f.write('{}\n'.format(mixing_rules[ff_type]))

def write_force_field(sigmas, epsilons, ff_type, workdir = '.'):
    # Guest-guest interactions are written to force_field.def
    with open(os.path.join(workdir, 'force_field.def'), 'w') as f:
        f.write('# number of rules to overwrite\n')
        f.write('0\n')
        f.write('# number of defined interactions\n')
        f.write('{}\n'.format(len(sigmas)))
        f.write('# {:19s} {:21s}    {:15s}    {:6s}  {:6s}\n'.format('type1', 'type2', 'interaction', 'epsilon', 'sigma'))
        for key in sorted(sigmas.keys(), key=lambda e: (e[0].split('_')[-1], e[1].split('_')[-1], e[0], e[1])):
            sigma = sigmas[key]
            epsilon = epsilons[key]
            if epsilon == 0.0:
                f.write('{:21s} {:21s}    {:15s}\n'.format(key[0], key[1], 'NONE'))
            else:
                potential, sigma, epsilon = convert(sigma, epsilon, ff_type)
                if potential == 'MM3_VDW': sigma = 2*sigma # MM3_VDW uses sum of vdw radii as parameter
                f.write('{:21s} {:21s}    {:15s}    {:6.3f}  {:6.3f}\n'.format(key[0], key[1], potential, epsilon, sigma))
        f.write('# number of mixing rules to overwrite\n')
        f.write('0\n')

def write_input(host, guest, framework, molecule, fn_out = 'simulation.input', workdir = '.',
                temp = 298*kelvin, press = 1*atm, rcut = 14.0*angstrom,
                nsteps = 1000000, nsteps_init = 0, print_step = 100, print_prop = 10000,
                restart = 'no', crash = 100):
    with open(os.path.join(workdir, fn_out), 'w') as f:
        # General block
        f.write('# General block\n')
        f.write('{:30s}    {}\n'.format('SimulationType', 'MonteCarlo'))
        f.write('{:30s}    {}\n'.format('NumberOfCycles', nsteps))
        f.write('{:30s}    {}\n'.format('NumberOfInitializationCycles', nsteps_init))
        f.write('{:30s}    {}\n'.format('RestartFile', restart))
        if crash > 0:
            f.write('{:30s}    {}\n'.format('ContinueAfterCrash', 'yes'))
            f.write('{:30s}    {}\n'.format('WriteBinaryRestartFileEvery', crash))
        f.write('{:30s}    {}\n'.format('PrintEvery', print_step))
        f.write('{:30s}    {}\n'.format('PrintPropertiesEvery', print_prop))
        f.write('\n')

        # ForceField block
        f.write('# ForceField block\n')
        f.write('{:50s}    {}\n'.format('ForceField', framework))
        f.write('{:50s}    {}\n'.format('UseChargesFromCIFFile', 'yes'))
        f.write('{:50s}    {}\n'.format('ChargeFromChargeEquilibration', 'no'))
        f.write('{:50s}    {}\n'.format('SymmetrizeFrameworkCharges', 'no'))
        f.write('{:50s}    {} # angstrom\n'.format('CutOff', rcut/angstrom))
        if all(guest.charges == 0.0):
            f.write('{:50s}    {}\n'.format('OmitAdsorbateAdsorbateCoulombInteractions', 'yes'))
        f.write('{:50s}    {}\n'.format('InternalFrameworkLennardJonesInteractions', 'no'))
        f.write('{:50s}    {}\n'.format('RemoveBondNeighboursFromLongRangeInteraction', 'yes'))
        f.write('{:50s}    {}\n'.format('RemoveBendNeighboursFromLongRangeInteraction', 'yes'))
        f.write('{:50s}    {}\n'.format('RemoveTorsionNeighboursFromLongRangeInteraction', 'yes'))
        f.write('\n')

        # Framework block
        f.write('# Framework block\n')
        f.write('Framework 0\n')
        f.write('{:30s}    {}\n'.format('FrameworkName', framework))
        rspacings = host.cell.rspacings
        f.write('{:30s}    {:.0f} {:.0f} {:.0f}\n'.format('UnitCells', *np.ceil(np.ones(3)*2*rcut/rspacings)))
        f.write('{:30s}    {}\n'.format('FlexibleFramework', 'no'))
        f.write('{:30s}    {:.0f} # kelvin\n'.format('ExternalTemperature', temp/kelvin))
        f.write('{:30s}    {:.0f} # pascal\n'.format('ExternalPressure', press/pascal))
        f.write('\n')

        # Molecule block
        f.write('# Molecule block\n')
        f.write('{:15s} {:31s} {}\n'.format('Component 0', 'MoleculeName', molecule))
        f.write('{:15s} {:31s} {}\n'.format('', 'MoleculeDefinition', ''))
        f.write('{:15s} {:31s} {}\n'.format('', 'Intra14VDWScalingValue', 0.0))
        f.write('{:15s} {:31s} {}\n'.format('', 'Intra14ChargeChargeScalingValue', 0.0))
        f.write('{:15s} {:31s} {}\n'.format('', 'TranslationProbability', 0.333333))
        f.write('{:15s} {:31s} {}\n'.format('', 'RotationProbability', 0.333333))
        f.write('{:15s} {:31s} {}\n'.format('', 'ReinsertionProbability', 0.333333))
        f.write('{:15s} {:31s} {}\n'.format('', 'SwapProbability', 1))
        f.write('{:15s} {:31s} {}\n'.format('', 'CreateNumberOfMolecules', 0))

    if crash > 0:
        with open(os.path.join(workdir, 'restart.input'), 'w') as f:
            f.write('{:30s}    {}\n'.format('ContinueAfterCrash', 'yes'))
            f.write('{:30s}    {}\n'.format('WriteBinaryRestartFileEvery', crash))


if __name__ == '__main__':
    # Input
    fn_host, fn_guest, fn_host_host, fn_host_guest, fn_guest_guest, temp, press = parse()
    host = System.from_file(fn_host)
    guest0 = System.from_file(fn_guest)
    guest1 = System.from_file(fn_guest)
    ff_host_host = ForceField.generate(host, fn_host_host)
    ff_guest_guest = ForceField.generate(guest0, fn_guest_guest)
    ff_host_guest = ForceField.generate(guest1, fn_host_guest)
    struct = os.path.basename(fn_host).split('.')[0]
    molecule = os.path.basename(fn_guest).split('.')[0]
    if not all(ff_host_guest.system.charges == ff_guest_guest.system.charges):
        if not all(ff_guest_guest.system.charges == 0.0):
            raise NotImplementedError('Guest has different charges in host-guest and guest-guest interaction')
    
    # Create workdir
    workdir = '{:.0f}K_{:.0f}Pa'.format(temp/kelvin, press/pascal)
    if not os.path.exists(workdir): os.mkdir(workdir)

    # Prepare framework and pseudo atoms
    write_cif(host, workdir = workdir, fn_out = '{}.cif'.format(struct))
    write_pseudo_atoms(ff_host_host, ff_host_guest, workdir = workdir)

    # Prepare host-guest interactions
    sigmas, epsilons, ff_type = read_host_guest(ff_host_host, ff_host_guest)
    write_force_field_mixing_rules(sigmas, epsilons, ff_type, workdir = workdir)

    # Prepare guest-guest interactions
    sig_cross, eps_cross, ff_type = read_guest_guest(ff_guest_guest)
    write_force_field(sig_cross, eps_cross, ff_type, workdir = workdir)

    # Prepare input-file
    write_input(host, guest0, struct, molecule, workdir = workdir, temp = temp, press = press)


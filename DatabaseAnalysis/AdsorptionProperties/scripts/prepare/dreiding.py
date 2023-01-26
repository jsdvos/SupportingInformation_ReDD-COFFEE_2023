import os

import time

from yaff import System, log
log.set_level(0)
from yaff.pes.parameters import ParameterDefinition, ParameterSection, Parameters
from yaff.pes.generator import *
from yaff.pes.generator import LJCrossGenerator, SquareOopDistGenerator
from molmod.units import angstrom, kcalmol
from molmod.periodic import periodic

class DREIDINGMachine():
    def __init__(self, system):
        self.system = system

    def get_pars():
        dreiding_pars = {}
        with open('dreiding.prm', 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:
                ffatype, r0, d0, dzeta = line.strip().split()
                r0 = float(r0)*angstrom
                d0 = float(d0)*kcalmol
                dzeta = float(dzeta)
                dreiding_pars[ffatype] = (r0, d0, dzeta)
        return dreiding_pars
    dreiding_pars = get_pars()

    def construct_lj(self):
        # Get parameters
        '''
            E_LJ(Yaff) = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
            Parameters: sigma, epsilon

        Within DREIDING, the nonbonded van der Waals interaction is described as

            E_LJ(DREIDING) = D0*((R0/r)**12 - 2*(R0/r)**6)
            Parameters: R0, D0
        
        The same expressions are found for
            sigma = 2**(-1/6)*R0
            epsilon = D0

        The mixing rules are the same (arimetic mean for sigma/R0,
        geometric mean for epsilon/D0)

        '''
        lj_pars = {}
        for i in range(self.system.natom):
            for ffatype in self.dreiding_pars.keys():
                if len(ffatype) > 3: continue # No implicit hydrogens or bridging hydrogens or hydrogen bonds
                element = ffatype[:2]
                if element[1] == '_': element = element[0]
                if self.system.numbers[i] == periodic[element].number:
                    r0, d0, dzeta = self.dreiding_pars[ffatype]
                    if self.system.get_ffatype(i) in lj_pars.keys():
                        assert (2**(-1.0/6.0)*r0, d0) == lj_pars[self.system.get_ffatype(i)]
                    else:
                        lj_pars[self.system.get_ffatype(i)] = (2**(-1.0/6.0)*r0, d0)
        # Generate Parameters object
        units = ParameterDefinition('UNIT', lines=[(-1, 'SIGMA A'), (-1, 'EPSILON kcalmol')])
        scale = ParameterDefinition('SCALE', lines=[(-1, '1 0.0'), (-1, '2 0.0'), (-1, '3 1.0')]) # 1-2 and 1-3 are excluded
        pars = ParameterDefinition('PARS', lines=[(-1, '{:10s} {:.4f} {:.4f}'.format(ffatype, lj_pars[ffatype][0]/angstrom, lj_pars[ffatype][1]/kcalmol)) for ffatype in self.system.ffatypes])
        return ParameterSection('LJ', definitions = {'UNIT': units, 'SCALE': scale, 'PARS': pars})

class Parameters(Parameters):
    def to_file(self, fn):
        '''
        Write the Parameters to a nice file format
        '''
        # Define implemented prefixes
        cov_prefixes = ['BONDHARM', 'BENDAHARM', 'BENDCHARM', 'BENDCOS', 'TORSION', 'TORSCPOLYSIX', 'OOPDIST', 'OOPCOS', 'OOPCHARM', 'CROSS']
        ei_prefixes = ['FIXQ']
        vdw_prefixes = ['LJ', 'LJCROSS', 'MM3']
        
        # Describe each prefix
        descriptions = {
'BONDHARM': '''# Bond stretch
# ============
#
# Mathematical form:
# E_BONDHARM = 0.5*K*(r - R0)**2
#

''',
'BENDAHARM': '''# Angle bend
# ==========
#
# Mathematical form:
# E_BENDAHARM = 0.5*K*(theta - THETA0)**2
#

''',
'BENDCHARM': '''# Angle bend
# ==========
#
# Mathematical form:
# E_BENDCHARM = 0.5*K*(cos(phi) - COS0)**2
#

''',
'BENDCOS': '''# Angle bend
# ==========
#
# Mathematical form:
# E_BENDCOS = 0.5*A*(1-cos(M*(phi - PHI0)))
#

''',
'TORSION': '''# Torsion
# =======
#
# Mathematical form:
# E_TORSION = 0.5*A*(1-cos(M*(phi - PHI0)))
#

''',
'TORSCPOLYSIX': '''# Torsion polysix term
# =======
#
# Mathematical form:
# E_TORSCPOLYSIX = C1*cos(phi) + C2*cos(phi)**2 + C3*cos(phi)**3 +
#                  C4*cos(phi)**4 + C5*cos(phi)**5 + C6*cos(phi)**6
#
# To ensure that the term has is even in phi (E(-phi) = E(phi)), all odd
# coefficients (C1, C3, C5) are put to zero.
#

''',
'OOPDIST': '''# Inversion
# =========
#
# Mathematical form:
# E_OOPDIST = 0.5*K*(d - D0)**2
#

''',
'SQOOPDIST': '''# Inversion
# =========
#
# Mathematical form:
# E_SQOOPDIST = 0.5*K*(d**2 - D0)**2
#
# WARNING: the D0 parameter is assumed to be the SQUARE of D0
# and has units of squared length [e.g. A**2]
#

''',
'OOPCOS': '''# Inversion
# =========
#
# Mathematical form:
# E_OOPCOS = 0.5*A*(1 - cos(phi))
#

''',
'OOPCHARM': '''# Inversion
# =========
#
# Mathematical form:
# E_OOPCHARM = 0.5*K*(cos(phi) - COS0)**2
#

''',
'CROSS': '''# Cross terms
# ===========
#
# Mathematical form:
# E_CROSS = KSS * (r0-R0) * (r1-R1)
#           + KBS0 * (r0 - R0) * (theta - THETA0)
#           + KBS1 * (r1 - R1) * (theta - THETA0)
#

''',
'FIXQ': '''# Electrostatic interactions
# ==========================
#
# Mathematical form:
# E_FIXQ = q_i*q_j/r*erf(r/R)
#
#       with q_i = q_0i + sum(p_ij, j bonded to i)
# 
# The total atomic point charge is given by the sum of the 
# pre-charge (q_0i) and the bond charge increments (p_ij).
# The error function is included to allow the charge to be 
# distributed according to a Gaussian distribution. By
# putting the charge radius R to zero, the classic point
# charge expression is obtained.
#

''',
'LJ': '''# Lennard-Jones potential
# =======================
#
# Mathematical form:
# E_LJ = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
#
#        with sigma    =  (sigma_i + sigma_j)/2
#             epsilon  =  sqrt(epsilon_i * epsilon_j)
#

''',
'LJCROSS': '''# Lennard-Jones potential
# =======================
#
# Mathematical form:
# E_LJCROSS = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
#

''',
'MM3': '''# MM3 variant of the Lennard-Jones potential
# ==========================================
#
# Mathematical form:
# E_MM3 = epsilon*(1.84e5 * exp(sigma/r) - 2.25*(sigma/r)**6)
#
#        with sigma    =  (sigma_i + sigma_j)/2
#             epsilon  =  sqrt(epsilon_i * epsilon_j)
#
# The ONLYPAULI parameter corresponds to an undocumented
# feature. Put it to 0 to get the original MM3 form.
#

'''
}

        # Load generators
        generators = {'BONDHARM': BondHarmGenerator,
                'BENDAHARM': BendAngleHarmGenerator,
                'BENDCHARM': BendCosHarmGenerator,
                'BENDCOS': BendCosGenerator,
                'TORSION': TorsionGenerator,
                'TORSCPOLYSIX': TorsionPolySixCosGenerator,
                'OOPDIST': OopDistGenerator,
                'SQOOPDIST': SquareOopDistGenerator,
                'OOPCOS': OopCosGenerator,
                'CROSS': CrossGenerator,
                'FIXQ': FixedChargeGenerator,
                'LJ': LJGenerator,
                'LJCROSS': LJCrossGenerator,
                'MM3': MM3Generator}
        
        def write_sections(prefixes, fn):
            # Write the sections in prefixes to the file
            with open(fn, 'w') as f:
                for prefix in prefixes:
                    section = self.sections.setdefault(prefix)
                    nlines = 0
                    if not section == None:
                        nlines = 0
                        for suffix in ['PARS', 'ATOM', 'BOND']:
                            definition = section.definitions.setdefault(suffix)
                            if not definition == None:
                                nlines += len(definition.lines)
                        if nlines == 0: continue
                        if not descriptions.setdefault(prefix) == None:
                            # Write description
                            f.write(descriptions[prefix])
                        for suffix in ['UNIT', 'SCALE', 'DIELECTRIC', 'PARS', 'ATOM', 'BOND']:
                            definition = section.definitions.setdefault(suffix)
                            if not definition == None:
                                if suffix in ['PARS', 'ATOM', 'BOND']:
                                    # Print header
                                    if prefix == 'LJ':
                                        nffatypes = 1
                                    elif prefix == 'LJCROSS':
                                        nffatypes = 2
                                    elif prefix == 'MM3':
                                        nffatypes = 1
                                    elif prefix == 'FIXQ' and suffix == 'ATOM':
                                        f.write('# Pre-charges (Q0) and charge radii (R)\n')
                                        nffatypes = 1
                                    elif prefix == 'FIXQ' and suffix == 'BOND':
                                        f.write('# Bond charge increments (P)\n')
                                        nffatypes = 2
                                    else:
                                        nffatypes = generators[prefix].nffatype
                                    npars = len(generators[prefix].par_info)
                                    if prefix == 'FIXQ' and suffix == 'ATOM':
                                        npars = 2
                                    if prefix == 'FIXQ' and suffix == 'BOND':
                                        npars = 1
                                    label = '# KEY'
                                    separation = '#'
                                    nchar = len(prefix + ':' + suffix) + 1
                                    while len(label) < nchar:
                                        label += ' '
                                    while len(separation) < nchar:
                                        separation += '-'
                                    for i in range(nffatypes):
                                        label += '{:21s} '.format('label' + str(i))
                                        separation += '-'*22
                                    for i in range(npars):
                                        if prefix == 'FIXQ' and suffix == 'ATOM':
                                            if i == 1:
                                                i = 2
                                        if prefix == 'FIXQ' and suffix == 'BOND':
                                            i = 1
                                        if generators[prefix].par_info[i][1] == int:
                                            label += '{:2s} '.format(generators[prefix].par_info[i][0])
                                            separation += '-'*3
                                        else:
                                            label += ' {:17s}'.format(generators[prefix].par_info[i][0])
                                            separation += '-'*18
                                    f.write('{}\n{}\n{}\n'.format(separation, label, separation))
                                for line in definition.lines:
                                    if suffix in ['PARS', 'ATOM', 'BOND']:
                                        words = line[1].split()
                                        ffatype_keys = words[:nffatypes]
                                        pars_keys = words[nffatypes:]
                                        new_line = ''
                                        for ffatype in ffatype_keys:
                                            new_line += '{:21s} '.format(ffatype)
                                        for i, parameter in enumerate(pars_keys):
                                            if prefix == 'FIXQ' and suffix == 'ATOM' and i == 1:
                                                i = 2
                                            if prefix == 'FIXQ' and suffix == 'BOND':
                                                i = 1
                                            if generators[prefix].par_info[i][1] == float:
                                                new_line += '{: .10e} '.format(float(parameter))
                                            elif generators[prefix].par_info[i][1] == int:
                                                new_line += '{:<2d} '.format(int(parameter))
                                            else:
                                                raise NotImplementedError('No formatting for parameter with type {} implemented'.format(generators[prefix].par_info[i][1]))
                                    else:
                                        new_line = line[1]
                                    f.write('{}:{} {}\n'.format(prefix, suffix, new_line))
                                f.write('\n')
                        f.write('\n\n')

        write_sections(cov_prefixes + ei_prefixes + vdw_prefixes, fn)

if __name__ == '__main__':
    def iter_todo():
        input_path = '../../data/input_files'
        for struct in os.listdir(input_path):
            fn_sys = os.path.join(input_path, struct, '{}_optimized.chk'.format(struct))
            fn_ei = os.path.join(input_path, struct, 'pars_cluster.txt')
            fn_out = os.path.join(input_path, struct, 'pars_noncov_dreiding.txt')
            yield fn_sys, fn_ei, fn_out

    # Run
    for fn_sys, fn_ei, fn_out in iter_todo():
        system = System.from_file(fn_sys)
        ff_machine = DREIDINGMachine(system)
        pars_lj = ff_machine.construct_lj()
        pars_ei = Parameters.from_file(fn_ei).sections['FIXQ']
        pars = Parameters(sections = {'FIXQ': pars_ei, 'LJ': pars_lj})
        pars.to_file(fn_out)


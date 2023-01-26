#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from yaff.pes.parameters import Parameters, ParameterSection, ParameterDefinition
from yaff.pes.generator import *
from yaff.pes.generator import LJCrossGenerator, SquareOopDistGenerator
from molmod.units import parse_unit

class ParametersCombination(Parameters):
    '''
    A member of the Parameters class that allows for combining different parameter files
    Currently only covalent and electrostatic (FIXQ) terms are supported
    '''
    @classmethod
    def load(cls, sbu, fns = None):
        if fns == None:
            fns = os.listdir('{}{}/{}/ff_pars'.format(sbu.sbu_path, sbu.name, sbu.termination))
        return cls.from_file(['{}{}/{}/ff_pars/{}'.format(sbu.sbu_path, sbu.name, sbu.termination, fn) for fn in fns])

    @classmethod
    def from_file(cls, fn):
        parameters = Parameters.from_file(fn)
        result = cls()
        for prefix, section in parameters.sections.items():
            result.sections[prefix] = ParameterSectionCombination(prefix, section.definitions, section.complain)
        return result

    def to_file(self, fn = None, fn_cov = None, fn_ei = None, fn_vdw = None):
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
                                    f.write('{}:{} {}\n'.format(prefix, suffix, line))
                                f.write('\n')
                        f.write('\n\n')

        if not fn == None and (fn_cov == fn_ei == fn_vdw == None):
            write_sections(cov_prefixes + ei_prefixes + vdw_prefixes, fn)
        else:
            if not fn_cov == None:
                write_sections(cov_prefixes, fn_cov)
            if not fn_ei == None:
                write_sections(ei_prefixes, fn_ei)
            if not fn_vdw == None:
                write_sections(vdw_prefixes, fn_vdw)


    def __str__(self):
        text = ''
        for prefix, section in self.sections.items():
            text += '{}\n'.format(section)
        return text

    def add_parameters(self, other, internal = True, mixed = False, termination = False):
        '''
        Add a Parameters object to this one.
        If a new section is encountered that is not present, it will be added.
        Only keys that are not yet present will be added to this section if it already exists.
        '''
        for other_prefix, other_section in other.sections.items():
            section = self.sections.get(other_prefix)
            if section == None:
                # Prefix not yet present in Parameters
                # Add the lines that are present in the given environment
                section = other_section.copy()
                section.reset_pars()
                for other_line in other_section.iter_pars_lines(internal, mixed, termination):
                    section.add_line(other_line)
                self.sections[other_prefix] = section
            else:
                section.add_section(other_section, internal, mixed, termination)

    def add_combination(self, other_parameters1, other_parameters2):
        '''
        Add a combination of two Parameters objects to this one.
        Both given Parameters object will be screened to detect overlapping terms and will mix them.
        '''
        for other_parameters in [other_parameters1, other_parameters2]:
            self.add_parameters(other_parameters, internal = True, mixed = False, termination = False)
        mixed_parameters = other_parameters1.get_mixed_parameters(other_parameters2)
        self.add_parameters(mixed_parameters)

    def get_mixed_parameters(self, other_parameters):
        '''
        Returns the mixed parameters of this Parameters object and another
        '''
        mixed_parameters = ParametersCombination()
        for prefix, section in self.sections.items():
            other_section = other_parameters.sections.get(prefix)
            if not other_section == None:
                mixed_section = section.get_mixed_section(other_section)
                mixed_parameters.sections[prefix] = mixed_section
        return mixed_parameters

    def get_yaff_parameters(self, mixed = False):
        '''
        Returns the yaff Parameters object, which is needed to initialize a ForceField object
        '''
        sections = {}
        for prefix, section in self.sections.items():
            sections[prefix] = section.get_yaff_section(mixed = mixed)
        return Parameters(sections)

    def copy(self):
        '''
        Create an independent ParametersCombination object
        '''
        sections = {}
        for prefix, section in self.sections.items():
            sections[prefix] = section.copy()
        return ParametersCombination(sections = sections)

    def compare(self, other_parameters, name, path = None):
        '''
        Compare the parameters with the parameters of the same system, but with a larger environment.
        The figures are put in the map 'path', with the name 'prefix'_'parametername'_'name'.pdf (Example: 'BONDHARM_K_PDBA.pdf')
        '''
        for prefix, section in self.sections.items():
            other_section = other_parameters.sections.get(prefix)
            if not other_section == None:
                section.compare(other_section, name, path = path)
            else:
                print('WARNING: section {} not recognized in the argument Parameters object'.format(prefix))

class ParameterSectionCombination(ParameterSection):
    '''
    A subclass of ParameterSection to allow control over it, so that two objects from this class can be combined.
    '''
    def __init__(self, prefix, definitions = None, complain = None):
        parameter_section = ParameterSection(prefix, definitions, complain)
        self.prefix = prefix
        self.definitions = {}
        self.complain = parameter_section.complain
        self.generator = None
        for x in globals().values():
            if isinstance(x, type) and issubclass(x, Generator) and x.prefix == prefix:
                self.generator = x()
        if self.generator == None:
            raise NotImplementedError('No generator with prefix {}'.format(prefix))
        for suffix, definition in parameter_section.definitions.items():
            self.definitions[suffix] = ParameterDefinitionCombination(self.generator, suffix, definition.lines, definition.complain)

    def __str__(self):
        text = ''
        for suffix, definition in self.definitions.items():
            text += '{}\n'.format(definition)
        return text

    def copy(self):
        '''
        Return an independent ParameterSectionCombination object
        '''
        definitions = {}
        for suffix, definition in self.definitions.items():
            lines = []
            for line in definition.lines:
                lines.append((-1, line.line))
            definitions[suffix] = ParameterDefinition(suffix, lines, definition.complain)
        return ParameterSectionCombination(self.prefix, definitions, self.complain)

    def iter_pars_lines(self, internal = True, mixed = True, termination = True):
        '''
        Iterate over the parameter lines (in the 'PARS' definition for covalent terms or 'ATOM' and 'BOND' for FIXQ terms).
        Different flags are present to control the outcome: allow internal terms, allow mixed terms (both internal and termination), allow termination terms
        '''
        if not self.prefix == 'FIXQ':
            for line in self.definitions['PARS'].lines:
                if line.get_flag(internal, mixed, termination):
                    yield line
        else:
            for line in self.definitions['ATOM'].lines:
                if line.get_flag(internal, mixed, termination):
                    yield line
            for line in self.definitions['BOND'].lines:
                if line.get_flag(internal, mixed, termination):
                    yield line

    def add_line(self, line):
        '''
        Add the line to the parameter lines
        '''
        if not self.prefix == 'FIXQ':
            self.definitions['PARS'].lines.append(line)
        else:
            if len(line.key) == 1:
                # Atomic parameters
                self.definitions['ATOM'].lines.append(line)
            else:
                # Bond parameters
                self.definitions['BOND'].lines.append(line)

    def add_section(self, other_section, internal, mixed, termination):
        '''
        For each line in the given section, adds it to this section if it is not yet present
        '''
        conversions = self.check_definitions(other_section)
        for other_line in other_section.iter_pars_lines(internal = internal, mixed = mixed, termination = termination):
            other_line = other_line.get_converted_line(conversions, self.generator)
            equiv_line = self.get_line(other_line)
            if equiv_line == None:
                self.add_line(other_line)
            else:
                if not equiv_line.pars == other_line.pars:
                    print('WARNING: I got the {} term {} two times with different parameters {} and {}. I keep the first one'.format(self.prefix, equiv_line.key, equiv_line.pars, other_line.pars))

    def get_mixed_section(self, other_section):
        '''
        Find the lines that can be mixed and returns a ParameterSectionCombination object of these mixed terms
        '''
        mixed_section = self.copy()
        mixed_section.reset_pars()
        conversions = self.check_definitions(other_section)
        for line in self.iter_pars_lines(internal = False, mixed = True, termination = False):
            for other_line in other_section.iter_pars_lines(internal = False, mixed = True, termination = False):
                other_line = other_line.get_converted_line(conversions, self.generator)
                mixed_line = ParameterLine.mix_lines(line, other_line)
                if not mixed_line == None:
                    mixed_section.add_line(mixed_line)
        return mixed_section

    def compare(self, other_section, name, path = None):
        '''
        Compare the parameters with the parameters of the same system, but with a larger environment.
        The figures are put in the map 'path', with the name 'prefix'_'parametername'_'name'.pdf (Example: 'BONDHARM_K_PDBA.pdf')
        '''
        def iter_parameter_pairs(line, other_line):
            if self.prefix == 'FIXQ':
                if len(line.key) == 1:
                    # Atomic parameters
                    par_info = self.generator.par_info[::2]
                else:
                    par_info = [self.generator.par_info[1]]
            else:
                par_info = self.generator.par_info
            for i in range(len(par_info)):
                yield par_info[i][0], line.pars[i], other_line.pars[i]
        conversions = self.check_definitions(other_section)
        parameters = {par_info[0]: [[], []] for par_info in self.generator.par_info}
        mixed_parameters = {par_info[0]: [[], []] for par_info in self.generator.par_info}
        fig_info = {par_info[0]: [[], []] for par_info in self.generator.par_info}
        for line in self.iter_pars_lines(internal = True, mixed = False, termination = False):
            other_line = other_section.get_line(line)
            other_line = other_line.get_converted_line(conversions, self.generator)
            for par_name, parameter, other_parameter in iter_parameter_pairs(line, other_line):
                parameters[par_name][0].append(parameter)
                parameters[par_name][1].append(other_parameter)
        for line in self.iter_pars_lines(internal = False, mixed = True, termination = False):
            other_line = other_section.get_termination_line(line)
            other_line = other_line.get_converted_line(conversions, self.generator)
            for par_name, parameter, other_parameter in iter_parameter_pairs(line, other_line):
                mixed_parameters[par_name][0].append(parameter)
                mixed_parameters[par_name][1].append(other_parameter)
        import matplotlib.pyplot as plt
        for par_info in self.generator.par_info:
            plt.figure()
            par_name = par_info[0]
            plt.title('{}: {} - {}'.format(name, self.prefix, par_name))
            for parameter_set, color, legend in [[parameters[par_name], 'C0', 'Internal term'], [mixed_parameters[par_name], 'C3', 'Mixed term']]:
                plt.scatter(parameter_set[0], parameter_set[1], c = color, label = legend)
            plt.legend(loc = 2)
            lims = plt.xlim()
            plt.plot(lims, lims, 'k--', linewidth = 0.7)
            plt.xlim(lims)
            plt.ylim(lims)
            unit = ''
            for line in self.definitions['UNIT']:
                if line.key[0] == par_name:
                    unit = line.pars[0]
            plt.xlabel('Aromatic termination [{}]'.format(unit))
            plt.ylabel('Larger termination [{}]'.format(unit))
            if path == None:
                plt.show()
            else:
                if not os.path.exists(path):
                    os.mkdir(path)
                plt.savefig(os.path.join(path, '{}_{}_{}.pdf'.format(self.prefix, par_name, name)), bbox_inches = 'tight')
                plt.savefig(os.path.join(path, '{}_{}_{}.png'.format(self.prefix, par_name, name)), bbox_inches = 'tight')

    def reset_pars(self):
        '''
        Removes all parameter lines in the section, but keep the UNIT, SCALE and DIELECTRIC definitions
        '''
        if not self.prefix == 'FIXQ':
            self.definitions['PARS'].lines = []
        else:
            self.definitions['ATOM'].lines = []
            self.definitions['BOND'].lines = []

    def check_definitions(self, other_section):
        '''
        Checks if two sections can be combined and returns the conversion between the different units
        '''
        def get_scale(definition):
            scale = [None, None, None, 1.0]
            for line in definition.lines:
                assert len(line.key) == 1
                assert len(line.pars) == 1
                scale[int(line.key[0])] = float(line.pars[0])
            return scale
        def get_dielectric(definition):
            assert len(definition.lines) == 1
            assert len(definition.lines[0].pars) == 1
            return float(definition.lines[0].pars[0])
        def get_unit(definition, name):
            for line in definition:
                assert len(line.key) == 1
                assert len(line.pars) == 1
                if line.key[0] == name:
                    return line.pars[0]
            return '1'
        if self.prefix == 'FIXQ':
            scale = get_scale(self.definitions['SCALE'])
            other_scale = get_scale(other_section.definitions['SCALE'])
            if not np.all(scale == other_scale):
                raise RuntimeError('Scalings are not the same, force fields can not be combined')
            dielectric = get_dielectric(self.definitions['DIELECTRIC'])
            other_dielectric = get_dielectric(other_section.definitions['DIELECTRIC'])
            if not dielectric == other_dielectric:
                raise RuntimeError('Dielectrics are not the same, force fields can not be combined')
        conversions = []
        par_info = self.generator.par_info
        for name, cls in par_info:
            unit = get_unit(self.definitions['UNIT'], name)
            other_unit = get_unit(other_section.definitions['UNIT'], name)
            conversions.append(parse_unit(other_unit)/parse_unit(unit))
        return conversions

    def line_already_present(self, other_line):
        '''
        Checks if a line is already present in this section
        '''
        int_flag, term_flag = other_line.get_environment()
        for line in self.iter_pars_lines(int_flag and not term_flag, int_flag and term_flag, term_flag and not int_flag):
            for equiv_key, equiv_pars in line.iter_equiv_key_pars():
                if equiv_key == other_line.key:
                    return True
        return False

    def get_line(self, other_line):
        '''
        Searches for the given line or an equivalent line
        '''
        int_flag, term_flag = other_line.get_environment()
        for line in self.iter_pars_lines(int_flag and not term_flag, int_flag and term_flag, term_flag and not int_flag):
            for equiv_key, equiv_pars in line.iter_equiv_key_pars():
                if equiv_key == other_line.key:
                    return ParameterLine.from_key_pars(list(equiv_key), list(equiv_pars), line.suffix, self.generator)

    def get_termination_line(self, other_line):
        '''
        Searches for the same line, but allows termination ffatypes in one line to be internal ffatypes in the other.
        '''
        def get_element(ffatype):
            if ffatype.split('_')[-1] == 'term':
                if ffatype[1].isdigit():
                    return ffatype[0]
                else:
                    return ffatype[:2]
            else:
                return ffatype.split('_')[0]
        for line in self.iter_pars_lines(internal = True, mixed = True, termination = False):
            for equiv_key, equiv_pars in line.iter_equiv_key_pars():
                flag = len(equiv_key) == len(other_line.key)
                if flag:
                    for ffatype1, ffatype2 in zip(other_line.key, equiv_key):
                        if ffatype1.split('_')[-1] == 'term' or ffatype2.split('_')[-1] == 'term':
                                if not get_element(ffatype1) == get_element(ffatype2):
                                    flag = False
                        else:
                            if not ffatype1 == ffatype2:
                                flag = False
                if flag:
                    return ParameterLine.from_key_pars(list(equiv_key), list(equiv_pars), line.suffix, self.generator)
        return None

    def get_yaff_section(self, mixed = False):
        '''
        Returns a Yaff ParameterSection object
        '''
        definitions = {}
        for suffix, definition in self.definitions.items():
            definitions[suffix] = definition.get_yaff_definition(mixed = mixed)
        return ParameterSection(self.prefix, definitions, self.complain)



class ParameterDefinitionCombination(ParameterDefinition):
    '''
    Subclass of ParameterDefinition, which contains ParameterLines objects instead of tuples with the line number and the line string.
    '''
    def __init__(self, generator, suffix, lines = None, complain = None):
        parameter_definition = ParameterDefinition(suffix, lines, complain)
        self.suffix = suffix
        self.generator = generator
        self.lines = []
        for line in parameter_definition.lines:
            self.lines.append(ParameterLine.from_line(line[1], suffix, generator))
        self.complain = parameter_definition.complain

    def __str__(self):
        text = ''
        for line in self.lines:
            text += '{}:{} {}\n'.format(self.generator.prefix, self.suffix, line)
        return text

    def get_yaff_definition(self, mixed = False):
        '''
        Returns a Yaff ParametersDefinition object
        '''
        lines = []
        for line in self.lines:
            if self.suffix in ['PARS', 'ATOM', 'BOND'] and mixed:
                if line.is_mixed():
                    lines.append((-1, line.line))
            else:
                lines.append((-1, line.line))
        result = ParameterDefinition(self.suffix, lines, self.complain)
        return result

class ParameterLine():
    def __init__(self, line, key, pars, suffix, generator):
        '''
        Each line of a Yaff Parameter file can be fitted in a ParameterLine object
        Arguments:
            line [str]: how the ParameterLine will be written out to a file
            key [list of str]: the keys of the line, which can contain parameter names, ffatypes or nothing (for DIELECTRIC)
            pars [list of str/float/int] the parameters of a line, which can be a unit [str], scale [float], dielectric or force field parameters
            generator [yaff Generator object]: the generator of the corresponding section

        TO DO: process lines so the line.line argument is written nicely in the output file
        '''
        self.line = line
        self.key = key
        self.pars = pars
        self.suffix = suffix
        self.generator = generator
        if self.suffix in ['PARS', 'ATOM', 'BOND']:
            self.line = ''
            for ffatype in key:
                self.line += '{:21s} '.format(ffatype)
            for i, parameter in enumerate(pars):
                if self.generator.prefix == 'FIXQ' and self.suffix == 'ATOM' and i == 1:
                    i = 2
                if self.generator.prefix == 'FIXQ' and self.suffix == 'BOND':
                    i = 1
                if generator.par_info[i][1] == float:
                    self.line += '{: .10e} '.format(parameter)
                elif generator.par_info[i][1] == int:
                    self.line += '{:<2d} '.format(parameter)
                else:
                    raise NotImplementedError('No formatting for parameter with type {} implemented'.format(generator.par_info[i][1]))


    @classmethod
    def from_line(cls, line, suffix, generator):
        '''
        Creates a ParamterLine from a string
        '''
        data = line.split()
        key = []
        pars = []
        if suffix in ['UNIT', 'SCALE']:
            key.append(data[0])
            pars.append(data[1])
        elif suffix in ['DIELECTRIC']:
            key.append('')
            pars.append(data[0])
        elif suffix in ['PARS', 'ATOM', 'BOND']:
            par_info = generator.par_info
            if generator.prefix == 'FIXQ':
                if suffix == 'ATOM':
                    # Atomic parameters
                    par_info = (par_info[0], par_info[2])
                elif suffix == 'BOND':
                    # Bond parameter
                    par_info = (par_info[1], )
            for element in data:
                type = par_info[len(pars)][1]
                try:
                    pars.append(type(element))
                except ValueError:
                    key.append(element)
        else:
            raise NotImplementedError('Suffix {} not recognized'.format(suffix))
        return cls(line, key, pars, suffix, generator)

    @classmethod
    def from_key_pars(cls, key, pars, suffix, generator):
        '''
        Create a ParameterLine from the key and pars
        '''
        line = ' '.join([str(element) for element in key + pars])
        return cls(line, key, pars, suffix, generator)

    def __str__(self):
        return self.line

    def iter_equiv_key_pars(self):
        '''
        Iterate over all equivalent keys and pars. This is only allowed for parameter lines ('PARS' for covalent terms, 'ATOM' or 'BOND' for FIXQ terms)

        TO DO: Only allow parameter lines
        '''
        if not self.generator.prefix == 'FIXQ':
            for equiv_key, equiv_pars in self.generator.iter_equiv_keys_and_pars(self.key, self.pars):
                yield equiv_key, equiv_pars
        else:
            if len(self.key) == 1:
                # Atomic parameters
                yield self.key, self.pars
            else:
                # Bond parameters
                yield self.key, self.pars
                yield self.key[::-1], (-self.pars[0],)

    @ classmethod
    def mix_lines(cls, line, other_line):
        '''
        Creates a new line that mixes two lines with parameters for overlapping terms
        If both terms cannot overlap, there is a None object returned

        The definition of overlapping terms follows the nomenclature of the ffatypes given to the SBU object:
            internal ffatypes: starts with element symbol, ends with SBU name, separated by '_'. In between more specific termination can be used.
                Example: 'C_B_PDBA'
            termination ffatypes: starts with the element symbol and the minimal number of bonds untill an internal atom is encountered. Ends with '_term'.
                Example: 'C3_term'

        Terms can only overlap if exactly one of the two i-th ffatypes is an internal ffatype and both are the same element.
        The resulting mix term consists only of internal ffatypes (in one of both SBUs) and the mixed parameters are a weighted average of the SBU
            parametes, where the weight is given according to the number of atoms present in each SBU.
        '''
        def get_elements(internal_ffatype, termination_ffatype):
            elements = []
            for ffatype in [internal_ffatype, termination_ffatype]:
                element = ffatype.split('_')[0]
                while len(element) > 0 and element[-1].isdigit():
                    element = element[:-1]
                elements.append(element)
            if internal_ffatype.endswith('-07') and termination_ffatype in ['N2_term', 'N2_H_term']:
                if termination_ffatype == 'N2_H_term':
                    elements[1] += '_H'
                else:
                    elements[1] += '_C'
                elements[0] += '_' + internal_ffatype.split('_')[1][0]
            return elements[0], elements[1]
        def get_mix_key_and_rescalings(key1, key2):
            if not len(key1) == len(key2):
                return None, None
            mix_key = []
            rescalings = np.array([0, 0])
            for ffatype1, ffatype2 in zip(key1, key2):
                if ffatype2.split('_')[-1] == 'term' and not ffatype1.split('_')[-1] == 'term':
                        internal_element, termination_element = get_elements(ffatype1, ffatype2)
                        if internal_element == termination_element:
                            mix_key.append(ffatype1)
                            rescalings[0] += 1
                        else:
                            return None, None
                elif ffatype1.split('_')[-1] == 'term' and not ffatype2.split('_')[-1] == 'term':
                        internal_element, termination_element = get_elements(ffatype2, ffatype1)
                        if internal_element == termination_element:
                            mix_key.append(ffatype2)
                            rescalings[1]+= 1
                        else:
                            return None, None
                else:
                    return None, None
            return mix_key, rescalings/float(len(mix_key))
        def get_mix_pars(pars1, pars2, rescalings):
            if not len(pars1) == len(pars2):
                return None
            mix_pars = []
            par_info = line.generator.par_info
            if line.generator.prefix == 'FIXQ':
                if len(pars1) == 2:
                    # Atomic parameters
                    par_info = par_info[::2]
                else:
                    # Bond parameters
                    par_info = [par_info[1]]
            for parameter1, parameter2 in zip(pars1, pars2):
                mix_parameter = parameter1 * rescalings[0] + parameter2 * rescalings[1]
                type = par_info[len(mix_pars)][1]
                mix_pars.append(type(mix_parameter))
            return mix_pars
        # Checks if the lines can be mixed
        # If not a None object is returned
        for equiv_key, equiv_pars in line.iter_equiv_key_pars():
            mix_key, rescalings = get_mix_key_and_rescalings(equiv_key, other_line.key)
            if not mix_key == None:
                mix_pars = get_mix_pars(equiv_pars, other_line.pars, rescalings)
                return cls.from_key_pars(mix_key, mix_pars, line.suffix, line.generator)
        return None

    def get_environment(self):
        '''
        Returns if the keys contain parameters in the internal of the SBU and in the termination of the SBU
        '''
        internal_flag = False
        termination_flag = False
        for ffatype in self.key:
            if ffatype.split('_')[-1] == 'term':
                termination_flag = True
            else:
                internal_flag = True
        return internal_flag, termination_flag

    def is_mixed(self):
        environment = None
        for ffatype in self.key:
            if environment == None:
                environment = ffatype.split('_')[-1]
            else:
                if not ffatype.split('_')[-1] == environment:
                    return True
        return False

    def get_flag(self, internal, mixed, termination):
        '''
        Returns the appropriate flag of the three given
        '''
        internal_flag, termination_flag = self.get_environment()
        if internal_flag and not termination_flag:
            return internal
        elif internal_flag and termination_flag:
            return mixed
        elif termination_flag and not internal_flag:
            return termination

    def get_converted_line(self, conversions, generator):
        '''
        Creates a converted line, where the parameters are converted according to the conversions argument
        '''
        par_info = generator.par_info
        key = []
        pars = []
        if generator.prefix == 'FIXQ':
            if len(self.key) == 1:
                # Atomic parameters
                conversions = conversions[::2]
                par_info = par_info[::2]
            else:
                # Bond parameters
                conversions = [conversions[1]]
                par_info = [par_info[1]]
        key = self.key
        for i in range(len(self.pars)):
            cls = par_info[i][1]
            pars.append(cls(conversions[i] * self.pars[i]))
        return self.from_key_pars(key, pars, self.suffix, generator)

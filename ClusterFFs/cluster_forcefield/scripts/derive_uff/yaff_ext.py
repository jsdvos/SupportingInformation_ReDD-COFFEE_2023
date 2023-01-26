from yaff.pes.parameters import Parameters

from yaff.pes.generator import Generator, ValenceGenerator, BondHarmGenerator, BendCosGenerator, BendCosHarmGenerator, TorsionGenerator, OopCosGenerator, LJGenerator, LJCrossGenerator
from yaff.pes.iclist import OopCos
from yaff.pes.vlist import Harmonic

class Parameters(Parameters):
    def to_file(self, fn):
        descriptions = {
'BONDHARM': '''# Bond stretch
# ============
#
# Mathematical form:
# E_BONDHARM = 0.5*K*(r - R0)**2
#

''',
'BENDAHARM': '''# Angle bend
# ============
#
# Mathematical form:
# E_BENDAHARM = 0.5*K*(theta - THETA0)**2
#

''',
'BENDCHARM': '''# Angle bend
# ============
#
# Mathematical form:
# E_BENDCHARM = 0.5*K*(cos(phi) - COS0)**2
#

''',
'BENDCOS': '''# Angle bend
# ============
#
# Mathematical form:
# E_BENDCOS = 0.5*A*(1-cos(M*(phi - PHI0)))
#

''',
'TORSION': '''# Torsion
# ============
#
# Mathematical form:
# E_TORSION = 0.5*A*(1-cos(M*(phi - PHI0)))
#

''',
'OOPDIST': '''# Inversion
# ============
#
# Mathematical form:
# E_OOPDIST = 0.5*K*(d - D0)**2
#

''',
'SQOOPDIST': '''# Inversion
# ============
#
# Mathematical form:
# E_SQOOPDIST = 0.5*K*(d**2 - D0)**2
#
# WARNING: the D0 parameter is assumed to be the SQUARE of D0
# and has units of squared length [e.g. A**2]
#

''',
'OOPCOS': '''# Inversion
# ============
#
# Mathematical form:
# E_OOPCOS = 0.5*A*(1 - cos(phi))
#

''',
'OOPCHARM': '''# Inversion
# ============
#
# Mathematical form:
# E_OOPCHARM = 0.5*K*(cos(phi) - COS0)**2
#

''',
'CROSS': '''# Cross terms
# ==============
#
# Mathematical form:
# E_CROSS = KSS * (r0-R0) * (r1-R1)
#           + KBS0 * (r0 - R0) * (theta - THETA0)
#           + KBS1 * (r1 - R1) * (theta - THETA0)
#

''',
'FIXQ': '''# Electrostatic interactions
# ======================
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
# ======================
#
# Mathematical form:
# E_LJ = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
#
#        with sigma    =  (sigma_i + sigma_j)/2
#             epsilon  =  sqrt(epsilon_i * epsilon_j)
#

''',
'LJCROSS': '''# Lennard-Jones potential
# ======================
#
# Mathematical form:
# E_LJCROSS = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
#

''',
'MM3': '''# MM3 variant of the Lennard-Jones potential
# ======================
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
        with open(fn, 'w') as f:
            generators = {}
            for x in globals().values():
                if isinstance(x, type) and issubclass(x, Generator):
                    if x.prefix in ['BONDHARM', 'BENDCOS', 'BENDCHARM', 'TORSION', 'OOPCOS', 'OOPCHARM', 'LJ', 'LJCROSS']:
                        generators[x.prefix] = x()
            for prefix in ['BONDHARM', 'BENDCOS', 'BENDCHARM', 'TORSION', 'OOPCOS', 'OOPCHARM', 'LJCROSS']:
                
                section = self.sections.setdefault(prefix)
                if not section == None and not len(section.definitions['PARS'].lines) == 0:
                    comment = None
                    if not descriptions.setdefault(prefix) == None:
                         comment = descriptions[prefix]
                    if not comment == None:
                        f.write(comment)
                    for suffix in ['UNIT', 'SCALE', 'DIELECTRIC', 'PARS', 'ATOM', 'BOND']:
                        definition = section.definitions.setdefault(suffix)
                        if not definition == None:
                            if suffix == 'PARS':
                                if prefix == 'LJ':
                                    nffatypes = 1
                                elif prefix == 'LJCROSS':
                                    nffatypes = 2
                                else:
                                    nffatypes = generators[prefix].nffatype
                                npars = len(generators[prefix].par_info)
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
                                    if generators[prefix].par_info[i][1] == int:
                                        label += '{:2s} '.format(generators[prefix].par_info[i][0])
                                        separation += '-'*3
                                    else:
                                        label += '{:16s} '.format(generators[prefix].par_info[i][0])
                                        separation += '-'*17
                                f.write('{}\n{}\n{}\n'.format(separation, label, separation))




                            for counter, line in definition.lines:
                                f.write('{}:{} {}\n'.format(prefix, suffix, line))
                            f.write('\n')
                    f.write('\n\n')


class OopCosHarmGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('K', float), ('COS0', float)]
    prefix = 'OOPCHARM'
    ICClass = OopCos
    VClass = Harmonic
    allow_super_position = True

    def iter_equiv_keys_and_pars(self, key, pars):
        # IC is the angle between the plane formed by i, j and k
        # and the bond between k and l
        # -> Both l and k have to be fixed
        yield key, pars
        yield (key[1], key[0], key[2], key[3]), pars

    def iter_indexes(self, system):
        # Loop over all atoms; if an atom has 3 neighbors,
        # it is candidate for an OopCos term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbors) == 3:
                # Yield a term for all three out-of-plane angles
                # with atom as center atom
                yield neighbours[0],neighbours[1],neighbours[2],atom
                yield neighbours[1],neighbours[2],neighbours[0],atom
                yield neighbours[2],neighbours[0],neighbours[1],atom

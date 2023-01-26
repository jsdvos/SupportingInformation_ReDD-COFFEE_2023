fn_uffpars = 'uff.prm'
import numpy as np
from itertools import permutations

from yaff import log
log.set_level(0)

from yaff.system import System
from yaff.pes.parameters import ParameterSection, ParameterDefinition
from yaff.pes.generator import *
from yaff.pes.generator import LJCrossGenerator
from molmod.periodic import periodic
from molmod.bonds import bonds
from molmod.molecular_graphs import MolecularGraph, NRingPattern, HasAtomNumber, HasNumNeighbors, HasNeighborNumbers, HasNeighbors
from molmod.graphs import CritAnd, CritOr, CritNot, CriteriaSet, GraphSearch
from molmod.transformations import Translation, Rotation, superpose, compute_rmsd
from molmod.units import deg, rad

from yaff_ext import OopCosHarmGenerator, Parameters

def normalize(v):
    return v/np.linalg.norm(v)

class UFFMachine():

    def __init__(self, system, hard = True, max_aromatic_cycle = 20, aromatic_rings = []):
        self.system = system
        if isinstance(system.bonds, type(None)):
            system.detect_bonds()
        self.graph = MolecularGraph(self.system.bonds, self.system.numbers)
        self.aromatics = set()
        self.aromatic_rings = []
        for ring in aromatic_rings:
            indices = set(ring)
            self.aromatic_rings.append(indices)
            self.aromatics.update(ring)
        self.geometries = []
        self._detect_aromatics(max_cycle = max_aromatic_cycle)
        self._detect_geometries()
        self._init_bond_orders()
        self._init_uff_ffatypes(hard = hard)

    def _init_bond_orders(self):
        if not hasattr(self, 'aromatics'):
            self._detect_aromatics()
        self.bond_orders = {}
        for i0, i1 in self.system.iter_bonds():
            number0 = self.system.numbers[i0]
            pos0 = self.system.pos[i0]
            number1 = self.system.numbers[i1]
            pos1 = self.system.pos[i1]
            bond_label = '-'.join(str(i) for i in sorted([i0, i1]))
            diff = pos1 - pos0
            self.system.cell.mic(diff)
            distance = np.linalg.norm(diff)
            bond_order = bonds.bonded(number0, number1, distance)
            if set([i0, i1]).issubset(self.aromatics):
                if any([set([i0, i1]).issubset(self.aromatic_rings[i]) for i in range(len(self.aromatic_rings))]):
                    bond_order = 1.5
            self.bond_orders[bond_label] = bond_order

    ############################     I/O    #################################  

    # UFF Parameters
    def get_pars():
        pars = {}
        with open(fn_uffpars, 'r') as f:
            for line in f:
                data = line.split()
                if len(data) > 0:
                    if data[0] == 'param':
                        ffatype = data[1]
                        pars[ffatype] = {'r1': float(data[2]),
                                        'theta0': float(data[3]),
                                        'x1': float(data[4]),
                                        'D1': float(data[5]),
                                        'zeta': float(data[6]),
                                        'Z1': float(data[7]),
                                        'Vi': float(data[8]),
                                        'Uj': float(data[9]),
                                        'Xi': float(data[10]),
                                        'Hard': float(data[11]),
                                        'Radius': float(data[12])}
        return pars
    uff_pars = get_pars()

    # Yaff generators
    def get_generators(self):
        generators = {}
        for x in globals().values():
            if isinstance(x, type) and issubclass(x, Generator):
                if x.prefix in ['BONDHARM', 'BENDCOS', 'BENDCHARM', 'TORSION', 'OOPCOS', 'OOPCHARM', 'LJ', 'LJCROSS']:
                    generators[x.prefix] = x()
        return generators

    def read_ffatypes(self, fn):
        if not self.system.ffatypes == None:
            print('WARNING: Overwriting ffatypes')
        ffatypes = [None]*self.system.natom
        with open(fn, 'r') as f:
            for i, line in enumerate(f.readlines()):
                ffatypes[i] = line.strip()
        self.system.ffatypes = np.array(ffatypes)

    def read_uff_ffatypes(self, fn):
        if not type(self.uff_ffatypes) == type(None):
            print('WARNING: Overwriting UFF ffatypes')
        uff_ffatypes = [None]*self.system.natom
        with open(fn, 'r') as f:
            for i, line in enumerate(f.readlines()):
                uff_ffatypes[i] = line.strip()
        self.uff_ffatypes = np.array(uff_ffatypes)

    def read_bond_orders(self, fn):
        if len(self.bond_orders) > 0:
            print('WARNING: Overwriting bond orders')
        with open(fn, 'r') as f:
            for line in f.readlines():
                bond_label, bond_order = line.strip().split()
                bond_order = float(bond_order)
                self.bond_orders[bond_label] = bond_order

    def write_uff_ffatypes(self, fn):
        with open(fn, 'w') as f:
            for uff_ffatype in self.uff_ffatypes:
                f.write('{}\n'.format(uff_ffatype))
    
    def write_bond_orders(self, fn):
        with open(fn, 'w') as f:
            for bond_label, bond_order in self.bond_orders.items():
                f.write('{} {}\n'.format(bond_label, bond_order))

    ########################    FFATYPE DETECTION    ########################

    # Aromatic cycles
    def _detect_aromatics(self, max_cycle = 20):
        '''
        Perform an initial scan to detect the aromatic atoms. This is done
        initially, because we only want to search for NRings once. The atom
        indices that indicate aromaticity are stored in self.aromatics.

        An aromatic ring is defined as a planar cycle with only B, C, N, O or
        S atoms that has 4n + 2 pi-electrons (Huckel's Rule). The pi-electrons
        are counted based on their element and environment:
        C: 1 pi-electron
        N (non-substituted): 1 pi-electron
        N (substituted): 2 pi-electrons
        O: 2 pi-electrons
        S: 2 pi-electrons
        B: 0 pi-electrons
        
        This has difficulties when metals are present.
        Limitation is in the detection of conjugated systems, which can be
        larger than only rings.
        '''
        def planar(indices, verbose = False):
            eps = 0.38
            indices = list(indices)
            if len(indices) <= 3:
                return True
            else:
                v0 = normalize(self.system.pos[indices[1]] - self.system.pos[indices[0]])
                i = 2
                flag = True
                while flag:
                    # Also cover case if colinear points are given
                    v1 = normalize(self.system.pos[indices[i]] - self.system.pos[indices[0]])
                    i += 1
                    flag = np.linalg.norm(v1 - v0) < 0.01 and i < len(indices)
                normal = normalize(np.cross(v0, v1))
                while i < len(indices):
                    v = self.system.pos[indices[i]] - self.system.pos[indices[0]]
                    i += 1
                    if verbose:
                        print(np.dot(v, normal))
                    if not abs(np.dot(v, normal)) < eps:
                        return False
                return True

        def huckel(indices):
            # Huckel's Rule: Cycles with 4n + 2 pi-electrons are aromatic rings
            pi_electrons = 0
            for k in indices:
                if self.system.numbers[k] == 7:
                    if len(self.system.neighs1[k]) == 2:
                        pi_electrons += 1
                    else:
                        pi_electrons += 2
                elif self.system.numbers[k] == 6:
                    pi_electrons += 1
                elif self.system.numbers[k] == 5:
                    pi_electrons += 0
                else:
                    assert self.system.numbers[k] in [8, 16]
                    pi_electrons += 2
            return (pi_electrons - 2)%4 == 0

        is_bcnos = CritOr(HasAtomNumber(5), HasAtomNumber(6), HasAtomNumber(7), HasAtomNumber(8), HasAtomNumber(16))
        criterium_bcnos = CriteriaSet(vertex_criteria = {0: is_bcnos, 1: is_bcnos})
        for i in range(3, max_cycle + 1):
            criterium_bcnos.vertex_criteria[i - 1] = is_bcnos
            pattern = NRingPattern(i, criteria_sets = [criterium_bcnos])
            gs = GraphSearch(pattern)
            for match in gs(self.graph):
                # Search for cycles with only C, N, O or S atoms
                indices = set(match.forward.values())
                if not indices in self.aromatic_rings and huckel(indices) and planar(indices):
                    self.aromatic_rings.append(indices)
                    self.aromatics.update(match.forward.values())

    # Geometries
    def _detect_geometries(self):
        '''
        Check which geometry returns the smallest RMSD.

        This has difficulties for the non-symmetric geometries where the center
        of mass is not in the origin.
        '''
        singular = [np.array([1.0, 0.0, 0.0])]
        linear_x = [np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0])]
        linear_y = [np.array([0.0, 1.0, 0.0]),
                np.array([0.0, -1.0, 0.0])]
        linear_z = [np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, -1.0])]
        square = linear_x + linear_y
        tshape = square[:3]
        octahedral = linear_x + linear_y + linear_z
        square_pyramidal = octahedral[:5]
        trigonal = [np.array([1.0, 0.0, 0.0]),
                np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3), 0.0]),
                np.array([np.cos(-2*np.pi/3), np.sin(-2*np.pi/3), 0.0])]
        angular_trigonal = trigonal[:2]
        trigonal_bipyramidal = trigonal + linear_z
        seesaw = trigonal_bipyramidal[1:]
        tetrahedral = [np.array([np.sqrt(8./9), 0.0, -1./3]),
                np.array([-np.sqrt(2./9), np.sqrt(2./3), -1./3]),
                np.array([-np.sqrt(2./9), -np.sqrt(2./3), -1./3]),
                np.array([0.0, 0.0, 1.0])]
        angular_tetrahedral = tetrahedral[:2]
        trigonal_pyramidal = tetrahedral[:3]
        geometry_names = {'singular': singular, 'linear': linear_x, 'trigonal': trigonal,
                'angular_trigonal': angular_trigonal, 'tetrahedral': tetrahedral,
                'angular_tetrahedral': angular_tetrahedral, 'trigonal_pyramidal': trigonal_pyramidal,
                'square': square, 'tshape': tshape, 'trigonal_bipyramidal': trigonal_bipyramidal,
                'seesaw': seesaw, 'octahedral': octahedral, 'square_pyramidal': square_pyramidal}
        geometry_indices = {'singular': 0, 'linear': 1, 'trigonal': 2, 'angular_trigonal': 2,
                'tetrahedral': 3, 'angular_tetrahedral': 3, 'trigonal_pyramidal': 3, 
                'square': 4, 'tshape': 4, 'trigonal_bipyramidal': 5, 'seesaw': 5,
                'octahedral': 6, 'square_pyramidal': 6}
        for i in range(self.system.natom):
            nei_pos = []
            for j in self.system.neighs1[i]:
                vec = self.system.pos[j]-self.system.pos[i]
                nei_pos.append(vec/np.linalg.norm(vec))
            min_rmsd = None
            min_name = None
            for name, geometry in geometry_names.items():
                if len(geometry) == len(nei_pos):
                    for comb_pos in permutations(nei_pos):
                        transformation = superpose(np.array(comb_pos), np.array(geometry))
                        rot = Rotation.from_matrix(transformation.matrix)
                        fit_neis = transformation * np.array(geometry)
                        rmsd = compute_rmsd(np.array(comb_pos), fit_neis)
                        if min_rmsd is None or rmsd < min_rmsd:
                            min_rmsd = rmsd
                            min_name = name
            if min_name is None:
                self.geometries.append('None')
            else:
                self.geometries.append(geometry_indices[min_name])

    # UFF FFatypes
    def _init_uff_ffatypes(self, hard = True):
        def is_aromatic(i, graph):
            return i in self.aromatics
        def is_singular(i, graph):
            return self.geometries[i] == 0
        def is_linear(i, graph):
            return self.geometries[i] == 1
        def is_trigonal(i, graph):
            return self.geometries[i] == 2
        def is_tetrahedral(i, graph):
            return self.geometries[i] == 3
        def is_square(i, graph):
            return self.geometries[i] == 4
        def is_trigonal_bipyramidal(i, graph):
            return self.geometries[i] == 5
        def is_octahedral(i, graph):
            return self.geometries[i] == 6

        if not hasattr(self, 'aromatics'):
            self._detect_aromatics()
        if not hasattr(self, 'geometries'):
            self._detect_geometries()
        # TO DO: Can you discern between multiple metal charges based on the coordination?
        # TO DO: fill in all elements
        filters = {'H_': CritAnd(HasAtomNumber(1), is_singular),
                'H_b': CritAnd(HasAtomNumber(1), CritNot(is_singular)),
                'He4+4': HasAtomNumber(2),
                'Li': HasAtomNumber(3),
                'Be3+2': HasAtomNumber(4),
                'B_3': CritAnd(HasAtomNumber(5), is_tetrahedral),
                'B_2': CritAnd(HasAtomNumber(5), is_trigonal),
                'C_3': CritAnd(HasAtomNumber(6), is_tetrahedral, CritNot(is_aromatic)),
                'C_R': CritAnd(HasAtomNumber(6), is_aromatic),
                'C_2': CritAnd(HasAtomNumber(6), is_trigonal, CritNot(is_aromatic)),
                'C_1': CritAnd(HasAtomNumber(6), is_linear, CritNot(is_aromatic)),
                'N_3': CritAnd(HasAtomNumber(7), is_tetrahedral, CritNot(is_aromatic)),
                'N_R': CritAnd(HasAtomNumber(7), is_aromatic),
                'N_2': CritAnd(HasAtomNumber(7), is_trigonal, CritNot(is_aromatic)),
                'N_1': CritAnd(HasAtomNumber(7), is_linear, CritNot(is_aromatic)),
                'O_3': CritAnd(HasAtomNumber(8), is_tetrahedral, CritNot(is_aromatic)),
                #'O_3_z'
                'O_R': CritAnd(HasAtomNumber(8), is_aromatic),
                'O_2': CritAnd(HasAtomNumber(8), CritOr(is_trigonal, is_singular), CritNot(is_aromatic)),
                'O_1': CritAnd(HasAtomNumber(8), is_linear, CritNot(is_aromatic)),
                'F_': HasAtomNumber(9),
                'Ne4+4': CritAnd(HasAtomNumber(10), is_square),
                'Na': HasAtomNumber(11),
                'Mg3+2': CritAnd(HasAtomNumber(12), is_tetrahedral),
                'Al3': CritAnd(HasAtomNumber(13), is_tetrahedral),
                'Al6+3': CritAnd(HasAtomNumber(13), is_octahedral),
                'Si3': CritAnd(HasAtomNumber(14), is_tetrahedral),
                #'P_3+3'
                #'P_3+5'
                #'P_3+q'
                #'S_3+2'
                #'S_3+4'
                #'S_3+6'
                'S_R': CritAnd(HasAtomNumber(16), is_aromatic),
                'S_2': CritAnd(HasAtomNumber(16), is_trigonal),
                'Cl': HasAtomNumber(17),
                'Ar4+4': CritAnd(HasAtomNumber(18), is_square),
                'K_': HasAtomNumber(19),
                'Ca6+2': CritAnd(HasAtomNumber(20), is_octahedral),
                'Sc3+3': CritAnd(HasAtomNumber(21), is_tetrahedral),
                'Ti3+4': CritAnd(HasAtomNumber(22), is_tetrahedral),
                'V_3+5': CritAnd(HasAtomNumber(23), is_tetrahedral),
                'Cr6+3': CritAnd(HasAtomNumber(24), is_octahedral),
                'Mn6+2': CritAnd(HasAtomNumber(25), is_octahedral),
                'Fe3+2': CritAnd(HasAtomNumber(26), is_tetrahedral),
                'Fe6+2': CritAnd(HasAtomNumber(26), is_octahedral),
                'Co6+3': CritAnd(HasAtomNumber(27), is_octahedral),
                'Ni4+2': CritAnd(HasAtomNumber(28), is_square),
                'Cu3+1': CritAnd(HasAtomNumber(29), is_tetrahedral),
                'Zn3+2': CritAnd(HasAtomNumber(30), is_tetrahedral),
                'Ga3+3': CritAnd(HasAtomNumber(31), is_tetrahedral),
                'Ge3': CritAnd(HasAtomNumber(32), is_tetrahedral),
                'As3+3': CritAnd(HasAtomNumber(33), is_tetrahedral),
                'Se3+2': CritAnd(HasAtomNumber(34), is_tetrahedral),
                'Br': HasAtomNumber(35),
                'Kr4+4': CritAnd(HasAtomNumber(36), is_square),
                'Rb': HasAtomNumber(37),
                'Sr6+2': CritAnd(HasAtomNumber(38), is_octahedral),
                'Y_3+3': CritAnd(HasAtomNumber(39), is_tetrahedral),
                'Zr3+4': CritAnd(HasAtomNumber(40), is_tetrahedral),
                'Zr8f4': CritAnd(HasAtomNumber(40), HasNumNeighbors(8)),
                'Nb3+5': CritAnd(HasAtomNumber(41), is_tetrahedral),
                'Mo6+6': CritAnd(HasAtomNumber(42), is_octahedral),
                'Mo3+6': CritAnd(HasAtomNumber(42), is_tetrahedral),
                'Tc6+5': CritAnd(HasAtomNumber(43), is_octahedral),
                
                'Lw6+3': CritAnd(HasAtomNumber(103), is_octahedral)}
        
        uff_ffatypes = self.apply_filters(filters, hard = hard)
        self.uff_ffatypes = np.array(uff_ffatypes)
    
    def overwrite_uff_ffatypes(self, filters):
        ffatypes = self.apply_filters(filters)
        for i in range(self.system.natom):
            if not ffatypes[i] == None:
                self.uff_ffatypes[i] = ffatypes[i]

    def apply_filters(self, filters, hard = False):
        '''
        Apply the atom type filters to the self.system. If hard is True,
        an error is thrown if no atom type is found
        '''
        ffatypes = []
        for i in range(self.system.natom):
            ffatype = None
            for label, atom_type in filters.items():
                if atom_type(i, self.graph):
                    if ffatype is None:
                        ffatype = label
                    else:
                        raise TypeError('Atom {} matches more than one type, at least {} and {}'.format(i, ffatype, label))
            if ffatype is None:
                if hard:
                    raise TypeError('Atom {} does not have any type'.format(i))
                else:
                    for label in filters.keys():
                        atom_type = label[:2].replace('_', '')
                        if atom_type == 'Lw': atom_type = 'Lr'
                        try:
                            if periodic[atom_type].number == self.system.numbers[i]:
                                ffatype = label
                                break
                        except:
                            print('atom symbol {} not recognized by molmod'.format(atom_type))
                    print('WARNING: couldnt find atom type for atom {}, assumed {}'.format(i, ffatype))
            ffatypes.append(ffatype)
        return ffatypes
        self.ffatypes = np.array(ffatypes)
   
    ############################    Parameter computation    #########################

    def add_to(self, lines, prefix, ffatype_key, ffatype_pars):
        generator = self.generators.setdefault(prefix)
        if generator == None:
            raise NotImplementedError('No generator with prefix {}'.format(prefix))
        for key, pars in generator.iter_equiv_keys_and_pars(ffatype_key, ffatype_pars):
            if key in lines.keys():
                for i in range(len(pars)):
                    assert abs(pars[i] - lines[key][i]) < 1e-3, 'Found different {} parameter sets for {}: {} and {}'.format(prefix, key, pars, lines[key])
                return
        lines[ffatype_key] = ffatype_pars
        return

    def to_pars_definition(self, lines, prefix):
        pars_def = ParameterDefinition('PARS')
        generator = self.generators.setdefault(prefix)
        if generator == None:
            raise NotImplementedError('No generator with prefix {}'.format(prefix))
        for key, pars in lines.items():
            ffatype_line = ''
            for ffatype in key:
                ffatype_line += '{:21s} '.format(ffatype)
            pars_line = ''
            for i, parameter in enumerate(pars):
                if generator.par_info[i][1] == float:
                    pars_line += '{:.10e} '.format(parameter)
                elif generator.par_info[i][1] == int:
                    pars_line += '{:<2d} '.format(parameter)
                else:
                    raise NotImplementedError('No formatting for parameter with type {} implemented'.format(generator.par_info[i][1]))
            pars_def.lines.append((-1, ffatype_line + pars_line))
        return pars_def

    def get_r(self, i, j):
        bond_label = '-'.join(str(i) for i in sorted([i, j]))
        bond_order = self.bond_orders.setdefault(bond_label)
        if bond_order == None:
            print('WARNING: No bond between atoms {} and {}'.format(i, j))
            return 0.0
        ffatype0 = self.uff_ffatypes[i]
        ffatype1 = self.uff_ffatypes[j]
        r_0 = self.uff_pars[ffatype0]['r1']
        r_1 = self.uff_pars[ffatype1]['r1']
        chi_0 = self.uff_pars[ffatype0]['Xi']
        chi_1 = self.uff_pars[ffatype1]['Xi']
        r_bo = -0.1332*(r_0 + r_1)*np.log(bond_order)
        r_en = r_0*r_1*(np.sqrt(chi_0) - np.sqrt(chi_1))**2/(chi_0*r_0 + chi_1*r_1)
        return r_0 + r_1 + r_bo - r_en

    # Bond
    def construct_bond(self):
        '''
        E_UFF = 1/2*K_ij*(r - r_ij)**2
        with r_ij = r_i + r_j + r_BO - r_EN
            r_BO = -0.1332*(r_i + r_j)*ln(n) (with bond order n)
            r_EN = r_i*r_j*(sqrt(chi_i) - sqrt(chi_j))**2/(chi_i*r_i + chi_j*r_j)
        with k_ij = 664.12*Z_i*Z_j/r_ij**3

        The bond order is set to 1.5 for aromatic rings

        ----- Yaff implementation -----

        E_BONDHARM = 1/2*K*(R - R0)**2
        Parameters:
            K = k_ij
            R0 = r_ij
        '''
        prefix = 'BONDHARM'
        units = ParameterDefinition('UNIT', lines=[(-1, 'K kcalmol/A**2'), (-1, 'R0 A')])
        ffatype_lines = {}
        for i, j in self.system.iter_bonds():
            # Get UFF atomtypes
            uff_i = self.uff_ffatypes[i]
            uff_j = self.uff_ffatypes[j]
            
            # Load parameters
            z_i = self.uff_pars[uff_i]['Z1']
            z_j = self.uff_pars[uff_j]['Z1']

            # Compute parameters
            r_ij = self.get_r(i, j)
            k_ij = 664.12*z_i*z_j/(r_ij**3)
            
            # Store parameters
            ffatype_key = tuple([self.system.get_ffatype(index) for index in [i, j]])
            ffatype_pars = (k_ij, r_ij)
            self.add_to(ffatype_lines, prefix, ffatype_key, ffatype_pars)
        pars = self.to_pars_definition(ffatype_lines, prefix)
        return ParameterSection(prefix, definitions = {'UNIT': units, 'PARS': pars})
    
    # Bend
    def construct_bend(self):
        '''
        If geometry of central atom is linear (n = 2), trigonal_planar (n = 3),
        square-planar (n = 4) or octahedral (n = 4):
                
                E_UFF = K_ijk/n**2*(1-cos(n*theta))

        Else:

                E_UFF = K_ijk/(2*sin(theta0)**2)*(cos(theta) - cos(theta0))**2

        with K_ijk = 664.12*Z_i*Z_j/r_ik**5*(3r_ij*r_jk*(1-cos(theta0)**2) - r_ik**2*cos(theta0))
            r_ik**2 = r_ij**2 + r_jk**2 - 2*r_ij*r_jk*cos(theta0)

        ----- Yaff implementation -----

        1) Specific case

        E_BENDCOS = 1/2*A*(1-cos(M*(phi - PHI0)))
        Parameters:
            M = n
            A = 2*K_ijk/n**2
            PHI0 = 0.0

        2) General case
        
        E_BENDCHARM = 1/2*K*(cos(phi) - COS0)**2
        Parameters:
            K = K_ijk/(sin(theta0))**2
            COS0 = cos(theta0)

        '''
        units_bendcos = ParameterDefinition('UNIT', lines=[(-1, 'A kcalmol'), (-1, 'PHI0 rad')])
        units_bendcharm = ParameterDefinition('UNIT', lines=[(-1, 'K kcalmol'), (-1, 'COS0 1')])
        ffatype_lines_bendcos = {}
        ffatype_lines_bendcharm = {}
        for i, j, k in self.system.iter_angles():
            # Get UFF atomtypes
            uff_i, uff_j, uff_k = (self.uff_ffatypes[index] for index in [i, j, k])
            
            # Load parameters
            theta0 = self.uff_pars[uff_j]['theta0']*deg/rad
            z_i = self.uff_pars[uff_i]['Z1']
            z_k = self.uff_pars[uff_k]['Z1']

            # Compute parameters
            geometry = uff_j[2]
            assert geometry in ['1', '2', '3', 'R', '4', '5', '6', '8']
            general = True
            if geometry == '1':
                # Linear
                n = 2
                general = False
            elif geometry in ['2', 'R']:
                # Trigonal
                n = 3
                general = False
            elif geometry in ['4', '6']:
                # Square or octahedral
                n = 4
                general = False

            r_ij = self.get_r(i, j)
            r_jk = self.get_r(j, k)
            r_ik = np.sqrt(r_ij**2 + r_jk**2 - 2*r_ij*r_jk*np.cos(theta0))
            k_ijk = 664.12*z_i*z_k/(r_ik**5)*(3*r_ij*r_jk*(1-np.cos(theta0)**2) - (r_ik**2)*np.cos(theta0))
            
            # Store parameters
            ffatype_key = tuple([self.system.get_ffatype(index) for index in [i, j, k]])
            if not general:
                prefix = 'BENDCOS'
                m = n
                a = 2*k_ijk/(n**2)
                phi0 = 0.0
                ffatype_pars = (m, a, phi0)
                self.add_to(ffatype_lines_bendcos, prefix, ffatype_key, ffatype_pars)
            else:
                prefix = 'BENDCHARM'
                ki = k_ijk/(np.sin(theta0)**2)
                c0 = np.cos(theta0)
                ffatype_pars = (ki, c0)
                self.add_to(ffatype_lines_bendcharm, 'BENDCHARM', ffatype_key, ffatype_pars)
        pars_def_bendcos = self.to_pars_definition(ffatype_lines_bendcos, 'BENDCOS')
        pars_def_bendcharm = self.to_pars_definition(ffatype_lines_bendcharm, 'BENDCHARM')
        pars_bendcos = ParameterSection('BENDCOS', definitions = {'UNIT': units_bendcos, 'PARS': pars_def_bendcos})
        pars_bendcharm = ParameterSection('BENDCHARM', definitions = {'UNIT': units_bendcharm, 'PARS': pars_def_bendcharm})
        return pars_bendcos, pars_bendcharm

    # Torsion
    def construct_torsion(self):
        '''
                E_UFF = V/2*(1-cos(n*phi0)cos(n*phi))

        Case 1) Both central atoms are sp3
            n = 3
            phi0 = pi
            V = sqrt(V_j*V_k)

            Exception 1: single bond between two sp3 atoms from group 6
                n = 2
                phi0 = pi/2
                V_j/k = 2 kcal/mol if atom j/k == O
                V_j/k = 6.8 kcal/mol if atom j/k != O
        
        Case 2) Both central atoms are sp2
            n = 2
            phi0 = pi
            V = 5*sqrt(U_j*U_k)*(1 + 4.18*len(n)) with bond order of central bond n

        Case 3) Mixed case: central atoms are sp2 and sp3
            n = 6
            phi0 = 0
            V = 1 kcal/mol

            Exception 2: single bond between sp3 atom from group 6 and sp2 atom not from group 6
                n = 2
                phi0 = pi/2
                V = 5*sqrt(U_j*U_k)*(1 + 4.18*len(n)) with bond order of central bond n

            Exception 3: single bond between sp3 atom and sp2 atom that is connected to another sp2 atom
                n = 3
                phi0 = pi
                V = 2 kcal/mol

        V should subsequently be divided by the number of torsions present about the central bond

        ----- Yaff implementation -----

        E_TORSION = 1/2*A*(1-cos(M*(phi - PHI0))) = 1/2*A*(1-cos(M*PHI0)*cos(M*phi)) if M*PHI0 = k*pi
        Parameters:
            M = n
            A = V
            PHI0 = phi0
        '''
        prefix = 'TORSION'
        units = ParameterDefinition('UNIT', lines=[(-1, 'A kcalmol'), (-1, 'PHI0 rad')])
        ffatype_lines = {}
        for i, j, k, l in self.system.iter_dihedrals():
            # Get UFF ffatypes
            uff_i, uff_j, uff_k, uff_l = (self.uff_ffatypes[index] for index in [i, j, k, l])
            
            # Load parameters
            v_j = self.uff_pars[uff_j]['Vi']
            v_k = self.uff_pars[uff_k]['Vi']
            u_j = self.uff_pars[uff_j]['Uj']
            u_k = self.uff_pars[uff_k]['Uj']
            bond_order = self.bond_orders['-'.join([str(index) for index in sorted([j, k])])]

            # Compute parameters
            n = None
            phi0 = None
            v = None
            if uff_i == 'H_':
                geom_i = '1'
            elif uff_i == 'F_':
                geom_i = '3'
            else:
                geom_i = uff_i[2]
            geom_j = uff_j[2]
            geom_k = uff_k[2]
            if uff_l == 'H_':
                geom_l = '1'
            elif uff_l == 'F_':
                geom_l = '3'
            else:
                geom_l = uff_l[2]
            assert geom_i in ['1', '2', '3', 'R', '4', '5', '6', '8']
            assert geom_j in ['1', '2', '3', 'R', '4', '5', '6', '8']
            assert geom_k in ['1', '2', '3', 'R', '4', '5', '6', '8']
            assert geom_l in ['1', '2', '3', 'R', '4', '5', '6', '8']
            if geom_i == 'R':
                geom_i = '2'
            if geom_j == 'R':
                geom_j = '2'
            if geom_k == 'R':
                geom_k = '2'
            if geom_l == 'R':
                geom_l = '2'
            if geom_j == '3' and geom_k == '3':
                # Both sp3 hybridisation
                exception = False
                if bond_order == 1.0 and self.system.numbers[j] in [8, 16, 34, 52, 84]:
                    exception = True
                    if self.system.numbers[j] == 8:
                        v_j = 2.0
                    else:
                        v_j = 6.8
                if bond_order == 1.0 and  self.system.numbers[k] in [8, 16, 34, 52, 84]:
                    exception = True
                    if self.system.numbers[k] == 8:
                        v_k = 2.0
                    else:
                        v_k = 6.8
                if not exception:
                    n = 3
                    phi0 = np.pi
                else:
                    n = 2
                    phi0 = np.pi/2
                v = np.sqrt(v_j*v_k)
            elif geom_j == '2' and geom_k == '2':
                # Both sp2 hybridisation
                n = 2
                phi0 = np.pi
                v = 5*np.sqrt(u_j*u_k)*(1+4.18*np.log(bond_order))
            elif set([geom_j, geom_k]) == set(['2', '3']):
                if geom_j == '3':
                    sp3 = j
                    sp2 = k
                else:
                    sp3 = k
                    sp2 = j
                exception2 = False
                exception3 = False
                if bond_order == 1.0:
                    group6 = [index for index in periodic.iter_numbers() if periodic[index].col == 16]
                    if self.system.numbers[sp3] in group6 and self.system.numbers[sp2] not in group6:
                        exception2 = True
                    if (geom_j == '2' and geom_i == '2') or (geom_k == '2' and geom_l == '2'):
                        exception3 = True
                if exception2:
                    n = 2
                    phi0 = np.pi/2
                    v = 5*np.sqrt(u_j*u_k)*(1+4.18*np.log(bond_order))
                elif exception3:
                    n = 3
                    phi0 = np.pi
                    v = 2
                else:
                    n = 6
                    phi0 = 0
                    v = 1
            if v is not None:
                v /= (len(self.system.neighs1[j]) - 1) * (len(self.system.neighs1[k]) - 1)

                # Store parameters
                ffatype_key = tuple([self.system.get_ffatype(index) for index in [i, j, k, l]])
                ffatype_pars = (n, v, phi0)
                self.add_to(ffatype_lines, prefix, ffatype_key, ffatype_pars)
        pars = self.to_pars_definition(ffatype_lines, prefix)
        return ParameterSection(prefix, definitions = {'UNIT': units, 'PARS': pars})

    # Inversion
    def construct_inversion(self):
        '''
        If the central atom is C_2, C_R, N_2, N_R, O_2, O_R:

                E_UFF = K_ijkl*(1-cos(omega))

        with K_ijkl = 50 kcal/mol if a central carbon atom is bonded to an O_2 atom and K_ijkl = 6 kcal/mol else
        Else
        
                E_UFF = K_ijkl/(1-cos(omega0))**2*(cos(omega) - cos(omega0))**2
        
        with omega0 depending on the atom number and K_ijkl = 22 kcal/mol

        K_ijkl should subsequently be divided by the number of inversions present about the central bond (i.e. three)

        ----- Yaff implementation -----

        1) Specific case:
        
        E_OOPCOS = 1/2*A*(1 - cos(phi))
        Parameters
            A = K_ijkl

        2) General case

        E_OOPCHARM = 1/2*K*(cos(omega) - COS0)**2
        Parameters
            K = K_ijkl/(1-cos(omega0))**2
            COS0 = cos(omega0)
        '''
        units_oopcos = ParameterDefinition('UNIT', lines=[(-1, 'A kcalmol')])
        units_oopcharm = ParameterDefinition('UNIT', lines=[(-1, 'K kcalmol'), (-1, 'COS0 1')])
        ffatype_lines_oopcos = {}
        ffatype_lines_oopcharm = {}
        for i0, i1, i2, i3 in self.system.iter_oops():
            # l is the central atom
            for i, j, k, l in [[i0, i1, i2, i3], [i2, i0, i1, i3], [i1, i2, i0, i3]]:
                # Get UFF ffatypes
                uff_i, uff_j, uff_k, uff_l = (self.uff_ffatypes[index] for index in [i, j, k, l])

                # Load parameters
                omega0 = {15: 84.4339*np.pi/180,
                        33: 86.9735*np.pi/180,
                        51: 87.7047*np.pi/180,
                        83: 90.0*np.pi/180}

                # Compute parameters
                if uff_l in ['C_2', 'C_R', 'N_2', 'N_R', 'O_2', 'O_R']:
                    # Specific case
                    general = False
                    if uff_l in ['C_2', 'C_R'] and 'O_2' in [uff_i, uff_j, uff_k]:
                        v = 50
                    else:
                        v = 6
                elif self.system.numbers[l] in omega0.keys():
                    # General case
                    general = True
                    v = 22
                    cos0 = np.cos(omega0[self.system.numbers[l]])

                else:
                    v = None
                if not v == None:
                    v /= 3

                # Store parameters
                ffatype_key = tuple([self.system.get_ffatype(index) for index in [i, j, k, l]])
                if not v == None:
                    if general:
                        prefix = 'OOPCHARM'
                        ffatype_pars = (v, cos0)
                        self.add_to(ffatype_lines_oopcharm, prefix, ffatype_key, ffatype_pars)
                    else:
                        prefix = 'OOPCOS'
                        ffatype_pars = (v, )
                        self.add_to(ffatype_lines_oopcos, prefix, ffatype_key, ffatype_pars)
        pars_def_oopcharm = self.to_pars_definition(ffatype_lines_oopcharm, 'OOPCHARM')
        pars_def_oopcos = self.to_pars_definition(ffatype_lines_oopcos, 'OOPCOS')
        pars_oopcharm = ParameterSection('OOPCHARM', definitions = {'UNIT': units_oopcharm, 'PARS': pars_def_oopcharm})
        pars_oopcos = ParameterSection('OOPCOS', definitions = {'UNIT': units_oopcos, 'PARS': pars_def_oopcos})
        if len(ffatype_lines_oopcharm) > 0:
            print('WARNING: OOPCHARM is used, which is not by default in Yaff!')
        return pars_oopcharm, pars_oopcos
            
    # Van der Waals
    def construct_lj(self):
        '''
        E_LJ(CROSS) = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
        Parameters: sigma, epsilon
        '''
        prefix = 'LJCROSS'
        units = ParameterDefinition('UNIT', lines=[(-1, 'SIGMA A'), (-1, 'EPSILON kcalmol')])
        scale = ParameterDefinition('SCALE', lines=[(-1, '1 0.0'), (-1, '2 0.0'), (-1, '3 1.0')]) # 1-2 and 1-3 are excluded
        ffatype_lines = {}
        for i in range(self.system.natom):
            for j in range(i, self.system.natom):
                # Get UFF ffatype
                uff_i, uff_j = (self.uff_ffatypes[index] for index in [i, j])

                # Load parameters
                x_i = self.uff_pars[uff_i]['x1']
                x_j = self.uff_pars[uff_j]['x1']
                epsilon_i = self.uff_pars[uff_i]['D1']
                epsilon_j = self.uff_pars[uff_j]['D1']

                # Compute parameters
                sigma_i = x_i/(2**(1./6))
                sigma_j = x_j/(2**(1./6))
                sigma = np.sqrt(sigma_i*sigma_j) # Geometric mean combination rule
                epsilon = np.sqrt(epsilon_i*epsilon_j) # Geometric mean combination rule
                
                # Store parameters
                ffatype_key = tuple([self.system.get_ffatype(index) for index in [i, j]])
                pars_key = (sigma, epsilon)
                self.add_to(ffatype_lines, prefix, ffatype_key, pars_key)
        pars = self.to_pars_definition(ffatype_lines, prefix)
        return ParameterSection(prefix, definitions = {'UNIT': units, 'SCALE': scale, 'PARS': pars}) 


    ##############################################################################
    ##############################################################################

    def build(self):
        if isinstance(self.system.ffatypes, type(None)):
            raise IOError('No ffatypes defined for system')
        self.generators = self.get_generators()
        pars_bondharm = self.construct_bond()
        pars_bendcos, pars_bendcharm = self.construct_bend()
        pars_torsion = self.construct_torsion()
        pars_oopcharm, pars_oopcos = self.construct_inversion()
        pars_lj = self.construct_lj()
        self.pars_cov = Parameters({'BONDHARM': pars_bondharm, 'BENDCOS': pars_bendcos, 
                            'BENDCHARM': pars_bendcharm, 'TORSION': pars_torsion,
                            'OOPCOS': pars_oopcos, 'OOPCHARM': pars_oopcharm})
        self.pars_lj = Parameters({'LJCROSS': pars_lj})



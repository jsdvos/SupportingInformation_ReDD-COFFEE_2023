import os
import itertools

import numpy as np
import time

from molmod.units import angstrom
from molmod.unit_cells import UnitCell
from copy import deepcopy

from SBU import SBU
from Topology import Topology, wyckoff_number_to_name, wyckoff_name_to_number, split_name
from Construct import GeometryConstructor

class Combination():
    def __init__(self, topology, sbus, name = None):
        self.topology = topology
        self.sbus = sbus
        self.constructor = None

    @classmethod
    def from_name(cls, name):
        data = name.split('_')
        top = data[0]
        topology = Topology.load(top)
        sbu_names = data[1:]
        if sbu_names[0] == '2':
            sbu_dict = {}
            for i in range(int(len(sbu_names)/2)):
                cn = sbu_names[2*i]
                sbu_name = sbu_names[2*i+1]
                sbu_dict[cn] = sbu_name
            sbu_names = []
            for i in range(len(topology.wyckoff_vertices)):
                wyckoff_name = wyckoff_number_to_name(i, False)
                wyckoff_vertex = topology.wyckoff_vertices[wyckoff_name]
                cn = wyckoff_vertex.cn
                sbu_names.append(sbu_dict[str(cn)])
            for i in range(len(topology.wyckoff_edges)):
                sbu_names.append(sbu_dict['2'])
        sbus = []
        for sbu_name in sbu_names:
            if sbu_name == 'None':
                sbus.append(None)
            else:
                core, reac, link = sbu_name.split('-')
                sbus.append(SBU.load(core, '_{}-{}'.format(reac, link)))
        return cls(topology, sbus, name)

    def get_name(self, long_name = False):
        name = '{}_'.format(self.topology.name)
        for sbu in self.sbus:
            if sbu == None:
                sbu_name = 'None'
            else:
                sbu_name = sbu.name + sbu.termination.replace('_', '-')
            name += '{}_'.format(sbu_name)
        if len(name) >= 242 and not long_name:
            # Name is to long for bash system
            # Assume that there are no mixed SBUs: all vertices with 
            # the same coordination number have the same SBUs
            sbus = {}
            for sbu in self.sbus:
                if sbu == None:
                    cn = 2
                    sbu_name = 'None'
                    assert sbus.get(cn) in [None, 'None']
                else:
                    cn = len(sbu.poes)
                    sbu_name = sbus.setdefault(cn)
                    if sbu_name == None:
                        sbu_name = sbu.name + sbu.termination.replace('_', '-')
                    else:
                        assert sbu_name == sbu.name + sbu.termination.replace('_', '-')
                sbus[cn] = sbu_name
            name = '{}_'.format(self.topology.name)
            for cn in sorted(sbus.keys()):
                name += '{}_{}_'.format(cn, sbus[cn])
        return name[:-1]

    def initialize_constructor(self, rescaling_tol = None):
        # If this returns a RuntimeError: rescaling failed
        self.constructor = GeometryConstructor(self.topology, self.sbus, rescaling_tol = rescaling_tol)

    def compute_rmsd(self):
        for node_name in self.constructor.sbus.keys():
            node = self.topology.get_node(node_name)
            self.constructor.reduce_configurations_geometric(node)
        return max(self.constructor.rmsd.values())

    def compute_properties(self):
        if self.constructor is None:
            self.initialize_constructor(rescaling_tol = 9999)
        
        # Rescaling and RMSD
        resc_factor = self.constructor.rescaling_factor
        resc_std = self.constructor.rescaling_std
        rmsd = self.compute_rmsd()

        # N_atom
        natom = 0
        for i in range(len(self.topology.wyckoff_vertices)):
            sbu = self.sbus[i]
            wyckoff_name = wyckoff_number_to_name(i)
            wyckoff_vertex = self.topology.wyckoff_vertices[wyckoff_name]
            if sbu is None:
                natom_sbu = 0
            else:
                natom_sbu = sbu.sys.natom
            natom += len(wyckoff_vertex)*natom_sbu
        for i in range(len(self.topology.wyckoff_edges)):
            sbu = self.sbus[len(self.topology.wyckoff_vertices) + i]
            wyckoff_name = wyckoff_number_to_name(i, True)
            wyckoff_edge = self.topology.wyckoff_edges[wyckoff_name]
            if sbu is None:
                natom_sbu = 0
            else:
                natom_sbu = sbu.sys.natom
            natom += len(wyckoff_edge)*natom_sbu

        # V_init
        lengths, angles = deepcopy(self.topology.unit_cell.parameters)
        for i in range(3):
            lengths[i] *= resc_factor
        if self.topology.dimension == '2D':
            lengths[2] = 20*angstrom # 1,1,2-supercell of 2D system with 10A c-vector
            natom *= 2 # 1,1,2-supercell
        volume = UnitCell.from_parameters3(lengths, angles).volume

        return resc_factor, resc_std, rmsd, natom, volume

    def build(self, mode = 'nucleation', fn = None):
        assert mode.lower() in ['random', 'nucleation'], 'The mode has to be random or nucleation, not {}'.format(mode)
        if mode.lower() == 'random':
            self.constructor.get_random_structure()
        if mode.lower() == 'nucleation':
            self.constructor.get_nucleation_structure()

class Database():
    def __init__(self, combinations):
        self.combinations = combinations

    @classmethod
    def from_file(cls, fn):
        '''
        A file with all combinations in the database given.
        Every line in this file represents a combinations.
        The first entry of the line should be the combination name.
        All other entries are neglected.
        '''
        combinations = []
        with open(fn, 'r') as f:
            for i, line in enumerate(f):
                try:
                    name = line.strip().split()[0]
                    combination = Combination.from_name(name)
                    combinations.append(combination)
                except:
                    print('WARNING: Could not read combination on line {} of {}, skipping'.format(i, fn))
        return cls(combinations)

    @classmethod
    def from_sbus_topologies(cls, sbus, topologies, new_sbus = None, max_wyckoff_sets = None, max_possibilities = 1000, mixed_linkers = None, mixed_vertex = None):
        if new_sbus == None:
            new_sbus = sbus
        all_combinations = []
        n_combinations = []
        sbus_cn = {2: [None]}
        for sbu in sbus:
            sbu_cn = sbus_cn.setdefault(len(sbu.poes), [])
            sbu_cn.append(sbu)
        for topology in topologies:
            n_vertex_mixed = 1
            n_edge_mixed = 1
            vertex_sbus = []
            edge_sbus = []
            vertex_cns = []
            
            for i in range(len(topology.wyckoff_vertices)):
                cn = topology.wyckoff_vertices[wyckoff_number_to_name(i)].cn
                vertex_cns.append(cn)
                sbu_cn = sbus_cn.setdefault(cn, [])
                n_vertex_mixed *= len(sbu_cn)
                vertex_sbus.append(sbu_cn)
            for i in range(len(topology.wyckoff_edges)):
                cn = topology.wyckoff_edges['_' + wyckoff_number_to_name(i)].cn
                sbu_cn = sbus_cn.setdefault(cn, [])
                n_edge_mixed *= len(sbu_cn)
                edge_sbus.append(sbu_cn)
            poss_vertex_mixed = itertools.product(*vertex_sbus)
            poss_edge_mixed = itertools.product(*edge_sbus)
            
            n_vertex_non_mixed = 1
            n_edge_non_mixed = 1
            for cn, sbu_cn in sbus_cn.items():
                if cn == 2:
                    n_edge_non_mixed *= len(sbu_cn)
                elif cn in vertex_cns:
                    n_vertex_non_mixed *= len(sbu_cn)
            poss_vertex_non_mixed = []
            index = {}
            i = 0
            for cn in sorted(sbus_cn.keys()):
                if cn in vertex_cns:
                    index[cn] = i
                    i += 1
            for poss_vertex_sbu in itertools.product(*(sbus_cn[cn] for cn in sorted(sbus_cn.keys()) if cn in vertex_cns)):
                poss_vertex_non_mixed.append(tuple([poss_vertex_sbu[index[cn]] for cn in vertex_cns]))
            poss_vertex_non_mixed = tuple(poss_vertex_non_mixed)
            poss_edge_non_mixed = (tuple([sbus_cn[2][i]])*len(edge_sbus) for i in range(len(sbus_cn[2])))
            
            
            
            n_vertex_mixed_edge_mixed = n_vertex_mixed*n_edge_mixed
            n_vertex_mixed_edge_non_mixed = n_vertex_mixed*n_edge_non_mixed
            n_vertex_non_mixed_edge_mixed = n_vertex_non_mixed*n_edge_mixed
            n_vertex_non_mixed_edge_non_mixed = n_vertex_non_mixed*n_edge_non_mixed
            n_combinations.append([n_vertex_mixed_edge_mixed, n_vertex_mixed_edge_non_mixed, n_vertex_non_mixed_edge_mixed, n_vertex_non_mixed_edge_non_mixed])

            iters = []
            all_mixed = False
            if 0 < n_vertex_mixed_edge_mixed <= max_possibilities:
                # Everything is feasible, only check this iter
                iter_sbus = itertools.product(poss_vertex_mixed, poss_edge_mixed)
                iters.append(iter_sbus)
                all_mixed = True
            if not all_mixed and 0 < n_vertex_mixed_edge_non_mixed <= max_possibilities:
                iter_sbus = itertools.product(poss_vertex_mixed, poss_edge_non_mixed)
                iters.append(iter_sbus)
            if not all_mixed and 0 < n_vertex_non_mixed_edge_mixed <= max_possibilities:
                iter_sbus = itertools.product(poss_vertex_non_mixed, poss_edge_mixed)
                iters.append(iter_sbus)
            if len(iters) == 0 and 0 < n_vertex_non_mixed_edge_non_mixed <= max_possibilities:
                # Only if none if the above is feasible
                iter_sbus = itertools.product(poss_vertex_non_mixed, poss_edge_non_mixed)
                iters.append(iter_sbus)

            combinations = []
            matching_combinations = []
            for iter_sbus in iters:
                for counter, possibility in enumerate(iter_sbus):
                    if any([sbu in new_sbus for sbu in possibility[0] + possibility[1]]):
                        text = ' '.join(sbu.name + sbu.termination for sbu in possibility[0])
                        for sbu in possibility[1]:
                            if sbu == None:
                                text += ' None'
                            else:
                                text += ' ' + sbu.name + sbu.termination
                        if text in combinations: continue # Combination already in other iter
                        combinations.append(text)
                        sbus_match = True
                        for i, edge_sbu in enumerate(possibility[1]):
                            wyckoff_edge = topology.wyckoff_edges[wyckoff_number_to_name(i, edge = True)]
                            neighbors = wyckoff_edge.nodes[0].neighbors
                            neighbor_wyckoff_indices = [wyckoff_name_to_number(split_name(neighbors[j].name)[1]) for j in range(2)]
                            neighbor_sbus = [possibility[0][j] for j in neighbor_wyckoff_indices]
                            if edge_sbu == None:
                                if not neighbor_sbus[0].match(neighbor_sbus[1]):
                                    sbus_match = False

                            else:
                                if not (neighbor_sbus[0].match(edge_sbu) and neighbor_sbus[1].match(edge_sbu)):
                                    sbus_match = False
                        if sbus_match:
                            matching_combinations.append(text)
                            combination = Combination(topology, possibility[0] + possibility[1])
                            all_combinations.append(combination)
        return cls(all_combinations)

    def to_file(self, fn):
        with open(fn, 'w') as f:
            for combination in self.combinations:
                combination_str = combination.topology.name + ' '
                for sbu in combination.sbus:
                    if sbu == None:
                        combination_str += 'None '
                    else:
                        combination_str += '{}{} '.format(sbu.name, sbu.termination)
                f.write(combination_str + '\n')

    def iter_combinations(self, ids = None):
        if ids == None:
            ids = range(len(self.combinations))
        copy_combinations = self.combinations[:]
        for index in ids:
            yield index, copy_combinations[index]

    def reduce(self, ids = None, resc_tol = None, rmsd_tol = None, natom_max = None, volume_max = None):
        for i, combination in self.iter_combinations(ids):
            resc_factor, resc_std, rmsd, natom, volume = combination.compute_properties()
            if resc_tol is not None:
                if resc_std > resc_tol:
                    self.combinations.remove(combination)
            if rmsd_tol is not None:
                if rmsd > rmsd_tol:
                    self.combination.remove(combination)
            if natom_max is not None:
                if natom > natom_max:
                    self.combination.remove(combination)
            if volume_max is not None:
                if volume > volume_max:
                    self.combination.remove(combination)

    def compute_rmsd(self, ids = None, fn = None):
        result = []
        if not fn == None:
            f = open(fn, 'w')
        for i, combination in self.iter_combinations(ids):
            combination.initialize_constructor(rescaling_tol = 9999)
            rmsd = combination.compute_rmsd()
            result.append(rmsd)
            if not fn == None:
                f.write('{} {}\n'.format(combination.get_name(), rmsd))
            combination.constructor = None
        if not fn == None:
            f.close()
        return result

    def compute(self, ids = None, fn = None):
        if not fn == None:
            f = open(fn, 'w')
        else:
            result = {}
        for i, combination in self.iter_combinations(ids):
            combination.initialize_constructor(rescaling_tol = 9999)
            resc_factor = combination.constructor.rescaling_factor
            resc_std = combination.constructor.rescaling_std
            rmsd = combination.compute_rmsd()
            combination.constructor = None
            if not fn == None:
                f.write('{} {} {} {}\n'.format(combination.get_name(), resc_factor, resc_std, rmsd))
            else:
                result[combination.get_name()] = [resc_factor, resc_std, rmsd]
        if not fn == None:
            return
        else:
            return result
    
    def compute_properties(self, ids = None, fn = None):
        if not fn is None:
            f = open(fn, 'w')
        else:
            result = {}
        for i, combination in self.iter_combinations(ids):
            name = combination.get_name()
            resc_factor, resc_std, rmsd, natom, volume = combination.compute_properties()
            if not fn is None:
                f.write('{} {} {} {} {} {}\n'.format(name, resc_factor, resc_std, rmsd, natom, volume))
            else:
                result[name] = (resc_factor, resc_std, rmsd, natom, volume)
        if not fn is None:
            return
        else:
            return result

    def build(self, folder, ids = None, resc_tol = None, mode = 'nucleation'):
        if not os.path.exists(folder):
            os.mkdir(folder)
        for i, combination in self.iter_combinations(ids):
            name = combination.get_name()
            print('Initializing constructor {}...'.format(name))
            combination.initialize_constructor(rescaling_tol = resc_tol)
            print('Computing RMSD of {}...'.format(name))
            combination.compute_rmsd()
            print('Building structure {}...'.format(name))
            combination.build(mode = mode)
            if not os.path.exists(os.path.join(folder, name)):
                os.mkdir(os.path.join(folder, name))
            combination.constructor.output(os.path.join(folder, name))
            combination.constructor = None # Reset to clear memory
                


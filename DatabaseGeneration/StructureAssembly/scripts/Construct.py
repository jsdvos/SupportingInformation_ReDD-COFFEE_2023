#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from copy import deepcopy

from yaff.pes.ff import ForceField
from yaff.pes.generator import FFArgs, apply_generators
from yaff.pes.parameters import Parameters
from yaff.sampling.dof import CartesianDOF
from yaff.sampling.opt import CGOptimizer
from yaff.sampling.io import XYZWriter
from yaff import System, log
from molmod.units import parse_unit, angstrom, kjmol, deg
from molmod.periodic import periodic
from molmod.molecular_graphs import MolecularGraph

from ParametersCombination import ParametersCombination
from Topology import Topology, split_name, wyckoff_name_to_number, get_diff
from SBU import SBU

class Cluster():
    '''
    A Cluster is a combination of SBUs. Compare it to a yaff System, where instead of atoms, SBUs are present

    Attributes:
        name [str]: name of the Cluster
        unit_cell [molmod UnitCell object]: UnitCell of the Cluster
        sbus [list of SBUs]: SBUs already present in the Cluster
        bonds [list of indices]: bonds between SBUs
        parameters [ParametersCombination object]: parameters of the final Cluster
        ff_sbus [list of str]: list to indicate whether the parameters of an SBU (with termination) are already added
        ff_bonds [list of list that contain two str]: list to indicate whether the mixed terms of two SBUs are already added
        ff [yaff ForceField object]: ForceField of the Cluster
        some atom attributes to create a yaff System..
    '''
    def __init__(self, sbus = None, unit_cell = None):
        self.unit_cell = unit_cell
        if sbus == None:
            sbus = []
        self.sbus = [] # Are filled during _init_atoms
        self.bonds = [] # Are filled during _init_atoms
        self.ff_sbus = []
        self.ff_bonds = []
        self.bond_ffs = {}
        self.parameters = ParametersCombination()
        self.uff_parameters = ParametersCombination()
        self.combine_parameters = True
        self.ff = None
        self._init_atoms(sbus)
        self.post_processed = False
        self.final_system = None

    def _init_atoms(self, sbus):
        self.atom_numbers = []
        self.atom_pos = []
        self.atom_ffatypes = []
        self.atom_bonds = []
        self.natoms = []
        self.natom = 0
        for sbu in sbus:
            self.add_sbu(sbu)

    def __str__(self):
        text = 'CLUSTER {} ({} SBUs)\n\n'.format(self.name, len(self.sbus))
        for i, sbu in enumerate(self.sbus):
            text += 'SBU {} - {} ({}-connected) {}\n'.format(i, sbu.name, len(sbu.poes), sbu.center)
        text += '\n'
        for i in range(self.natom):
            text += 'ATOM {} - {} ({}): {}\n'.format(i, periodic[self.atom_numbers[i]].symbol, self.atom_ffatypes[i], self.atom_pos[i])
        return text

    def add_sbu(self, sbu):
        '''
        Add an SBU to the Cluster
        '''
        self.sbus.append(sbu)
        for i in range(sbu.sys.natom):
            self.atom_numbers.append(sbu.sys.numbers[i])
            self.atom_pos.append(sbu.sys.pos[i])
            self.atom_ffatypes.append(sbu.sys.ffatypes[sbu.sys.ffatype_ids[i]])
        for bond in sbu.sys.bonds:
            self.atom_bonds.append([bond[0] + self.natom, bond[1] + self.natom])
        for poe in sbu.poes:
            poe[1] += self.natom
        self.natoms.append(sbu.sys.natom)
        self.natom = sum(self.natoms)
        if self.combine_parameters and not sbu.name + sbu.termination in self.ff_sbus:
            self.parameters.add_parameters(sbu.parameters, internal = True, mixed = False, termination = False)
            self.uff_parameters.add_parameters(sbu.uff_parameters, internal = True, mixed = False, termination = False)
            self.ff_sbus.append(sbu.name + sbu.termination)
        self.ff = None

    def remove_sbu(self, sbu):
        '''
        Remove an SBU from the Cluster.

        The parameters of the SBU remain in the cluster parameters attribute as there can be other SBUs present in the Cluster
        '''
        if not sbu in self.sbus:
            print('Remark: SBU {} is not in the cluster, and cannot be removed'.format(sbu.name))
        else:
            index = self.sbus.index(sbu)
            first_index = sum(self.natoms[:index])
            last_index = sum(self.natoms[:index+1]) - 1
            atom_indices = list(range(first_index, last_index + 1))
            for i in atom_indices[::-1]:
                del self.atom_numbers[i]
                del self.atom_pos[i]
                del self.atom_ffatypes[i]
            for i in range(len(self.atom_bonds) - 1, -1, -1):
                atom_bond = self.atom_bonds[i]
                if atom_bond[0] in atom_indices or atom_bond[1] in atom_indices:
                    del self.atom_bonds[i]
            for i in range(len(self.bonds) - 1, -1, -1):
                if index in self.bonds[i]:
                    del self.bonds[i]
            for i in range(index + 1, len(self.sbus)):
                for poe in self.sbus[i].poes:
                    poe[1] -= sbu.sys.natom
            for poe in sbu.poes:
                poe[1] -= first_index
                poe[2] = None
            del self.natoms[index]
            self.natom -= sbu.sys.natom
            self.sbus.remove(sbu)
        self.ff = None

    def update_unit_cell(self, unit_cell):
        '''
        Updates the unit cell of the Cluster
        '''
        self.unit_cell = unit_cell

    def get_atom_sys(self):
        '''
        Returns the yaff System object from the Cluster
        '''
        if self.unit_cell == None:
            return System(np.array(self.atom_numbers), np.array(self.atom_pos), ffatypes = np.array(self.atom_ffatypes), bonds = np.array(self.atom_bonds))
        else:
            rvecs = []
            for i in range(3):
                rvecs.append(self.unit_cell.matrix[:, i])
            return System(np.array(self.atom_numbers), np.array(self.atom_pos), ffatypes = np.array(self.atom_ffatypes), bonds = np.array(self.atom_bonds), rvecs = np.array(rvecs))

    def get_bond_energy(self, atom_bond, **kwargs):
        '''
        Get the covalent energy that is associated with a bond between 2 different
        SBUs. As all internal coordinates of the System remain fixed, but the bond
        between the two atoms associated with the points of extension, this can be
        used to determine the covalent energy difference between multiple configurations
        in a much more efficient way than computing energy of the whole System.
        '''
        if self.bond_ffs.get(atom_bond) == None or None:
            sys, new_to_old_indices = self.get_bond_sys(atom_bond)
            ff_args = FFArgs(**kwargs)
            apply_generators(sys, self.parameters.get_yaff_parameters(mixed = False), ff_args)
            ff = ForceField(sys, ff_args.parts, ff_args.nlist)
            self.bond_ffs[atom_bond] = [ff, new_to_old_indices]
        else:
            ff, new_to_old_indices = self.bond_ffs.get(atom_bond)
            new_pos = []
            for i in range(ff.system.natom):
                new_pos.append(self.atom_pos[new_to_old_indices[i]])
            ff.update_pos(np.array(new_pos))
        ff.part_valence.compute()
        bond_cov_energy = ff.part_valence.energy
        return bond_cov_energy

    def get_bond_sys(self, atom_bond):
        '''
        Returns a System containing only the atoms that are needed to construct
        the internal coordinates that include the given bond. These are all atoms
        that are separated by at most 2 bonds from the central bond atoms.

        Returns also a dictionary that maps the new indices onto the old indices
        of the whole System.
        '''
        atom0, atom1 = atom_bond
        new_atom_numbers = []
        new_atom_pos = []
        new_atom_ffatypes = []
        new_atom_bonds = []
        old_to_new_indices = {}
        new_to_old_indices = {}
        natom = 0
        for index in atom0, atom1:
            new_atom_numbers.append(self.atom_numbers[index])
            new_atom_pos.append(self.atom_pos[index])
            new_atom_ffatypes.append(self.atom_ffatypes[index])
            old_to_new_indices[index] = natom
            new_to_old_indices[natom] = index
            natom += 1
        new_atom_bonds.append([0, 1])
        sys = System(np.array(new_atom_numbers), np.array(new_atom_pos), ffatypes = np.array(new_atom_ffatypes), bonds = np.array(new_atom_bonds))
        for i in range(2):
            start_natom = natom
            for old_index0, old_index1 in self.atom_bonds:
                for old_index, other_old_index in [[old_index0, old_index1], [old_index1, old_index0]]:
                    if old_index in old_to_new_indices and old_to_new_indices[old_index] < start_natom:
                        if not other_old_index in old_to_new_indices:
                            new_atom_numbers.append(self.atom_numbers[other_old_index])
                            new_atom_pos.append(self.atom_pos[other_old_index])
                            new_atom_ffatypes.append(self.atom_ffatypes[other_old_index])
                            old_to_new_indices[other_old_index] = natom
                            new_to_old_indices[natom] = other_old_index
                            natom += 1
                            new_atom_bonds.append([old_to_new_indices[old_index], old_to_new_indices[other_old_index]])
        rvecs = []
        for i in range(3):
            rvecs.append(self.unit_cell.matrix[:, i])
        sys = System(np.array(new_atom_numbers), np.array(new_atom_pos), ffatypes = np.array(new_atom_ffatypes), bonds = np.array(new_atom_bonds), rvecs = np.array(rvecs))
        return sys, new_to_old_indices

    def to_chk(self, fn = None):
        '''
        Write the System to a chk file
        '''
        self.to_file(fn, '.chk')
        print('Printed Cluster to CHK-file {}'.format(fn))

    def to_file(self, fn, extension):
        '''
        General method to write the System to a file
        '''
        if not fn == None:
            if not fn.endswith(extension):
                if '.' in fn.split('/')[-1]:
                    raise RuntimeError('File name {} already has an extension which is not {}'.format(fn, extension))
                fn = fn + extension
        else:
            fn = self.name + extension
        sys = self.get_atom_sys()
        sys.to_file(fn)
        if self.post_processed:
            final_parts = fn.split('.')
            final_parts[-2] += '_final'
            final_fn = '.'.join(final_parts)
            self.final_system.to_file(final_fn)

class GeometryConstructor():
    '''
    Workplace where all methods to create a new geometry starting from a Topology and a number of SBUs are implemented

    Attributes:
        name [str]: name of the GeometryConstructor
        topology [Topology]: Topology of the final structure
        sbus [dict of node_name: SBU]: Dictionary containing all SBUs in the structure
        cluster [Cluster]: Cluster object to put the SBUs in
        combinations [dict of node_name: neighbor combinations]: Dictionary containing all neighbor combinations of a given node with the smallest RMSD
        final_combinations [dict of node_name: neighbor combination]: Dictionary containing the neighbor combination that is actually applied
        rmsd [dict of node_name: RMSD]: Dictionary containing all RMSD to quantify the mismatch between the node and its SBU

    The general workflow is as follows:
        1) Rescale the unit cell of the topology and the cluster
        2) Put the center of all SBUs on their place
        3) Get all possible neighbor combinations with the smallest RMSD
        4) Choose a suitable neighbor combination for each node and build the final structure

    Step 1-3 are general steps that are the same for every top-down building method
    In step 4 it can be chosen how the final neighbor combination is chosen. Currently, two algorithms are implementd:

    Random build:
        4a) For every node, choose a random neighbor combination
        4b) Add the SBU in the chosen neighbor combination to the Cluster

    Nucleation build:
        4a) Iterate breadth first through the Topology Graph
        4b) For every node encountered, choose the neighbor combination that minimizes the covalent energy
        4c) Add the SBU in the chosen neighbor combination to the Cluster
    '''

    def __init__(self, topology, sbus, rescaling_tol = 0.1):
        self.topology = topology.copy()
        self.rescaling_factor = None
        self.rescaling_std = None
        self._init_sbus(sbus)
        self.cluster = Cluster(unit_cell = self.topology.unit_cell)
        self.configurations = {node_name: [] for node_name in self.sbus.keys()}
        self.configurations_data = {node_name: {} for node_name in self.sbus.keys()}
        self.final_configurations = {node_name: None for node_name in self.sbus.keys()}
        self.rmsd = {node_name: None for node_name in self.sbus.keys()}
        self._rescale(tol = rescaling_tol)
        self._translate_sbus()
        self._init_configurations()
        self.name = '{}_'.format(topology.name)
        for sbu in sbus:
            if sbu == None:
                self.name += 'None_'
            else:
                self.name += '{}_'.format(sbu.name + sbu.termination)
        self.name = self.name[:-1]

    def _init_sbus(self, sbus):
        if not len(sbus) == len(self.topology.wyckoff_vertices) + len(self.topology.wyckoff_edges):
            raise RuntimeError('Topology {} expected {} SBUS, but got {}'.format(self.topology.name, len(self.topology.wyckoff_vertices) + len(self.topology.wyckoff_edges), len(sbus)))
        self.sbus = {}
        for node in self.topology.iter_nodes():
            prefix, wyckoff_name, number, suffix = split_name(node.name)
            wyckoff_number = wyckoff_name_to_number(wyckoff_name)
            if prefix == '':
                sbu = sbus[wyckoff_number]
            else:
                sbu = sbus[len(self.topology.wyckoff_vertices) + wyckoff_number]
            if not sbu == None:
                self.sbus[node.name] = sbu.copy()

    def _rescale(self, tol = None):
        '''
        1) Rescale the unit cell of the topology and the cluster

        For every edge, a rescaling factor can be defined so that the length
        between the neighboring vertices in the topology is the same as the length
        between the center of the SBUs placed on those vertices. As each edge in
        the same WyckoffSet connects the same SBUs, these rescaling factors will
        be the same.

        The final rescaling factor is the mean of the rescaling factors found for
        every WyckoffSet. If the standard deviation of these rescaling factors
        is larger than the given threshold delta, the SBUs are not considered as
        fitted for the Topology.
        '''
        if tol == None:
            tol = 0.1
        factors = []
        for wyckoff_edge_name, wyckoff_edge in self.topology.wyckoff_edges.items():
            wyckoff_factors = []
            for edge in wyckoff_edge.nodes:
                neighbor1, neighbor2 = edge.neighbors
                prefix1, wyckoff_name1, number1, suffix1 = split_name(neighbor1.name)
                prefix2, wyckoff_name2, number2, suffix2 = split_name(neighbor2.name)
                sbu1 = self.sbus[prefix1 + wyckoff_name1 + str(number1)] # Possibly outside unit cell -> suffix not used
                sbu2 = self.sbus[prefix2 + wyckoff_name2 + str(number2)] # Possibly outside unit cell -> suffix not used
                r1 = sbu1.get_radius()
                r2 = sbu2.get_radius()
                sbu_edge = self.sbus.get(edge.name)
                if sbu_edge == None:
                    r_edge = 0
                else:
                    r_edge = sbu_edge.get_radius()
                distance_vertices = np.linalg.norm(self.topology.unit_cell.to_cartesian(neighbor2.frac_pos) - self.topology.unit_cell.to_cartesian(neighbor1.frac_pos))
                distance_sbus = r1 + 2* r_edge + r2
                wyckoff_factors.append(distance_sbus/distance_vertices)
            factors.append(np.mean(wyckoff_factors))
        if np.std(factors) > tol:
            raise RuntimeError('Topology {} cannot be rescaled so that the SBUs {} fit on it'.format(self.topology.name, ', '.join('{} -> {}'.format(node, sbu.name) for node, sbu in self.sbus.items())))
        self.rescaling_factor = np.mean(factors)
        self.rescaling_std = np.std(factors)
        self.topology.rescale(self.rescaling_factor)
        self.cluster.update_unit_cell(self.topology.unit_cell)

    def _translate_sbus(self):
        '''
        2) Put the center of all SBUs on their place

        Translate every SBU to the position of the node it is putted on.

        The edges have to get an additional translation because the center of
        their SBU is not always exactly in the middle of the centers of both
        neighboring vertices. This happens whenever the radii of the neighboring
        SBUs is not the same.
        '''
        for node in self.topology.iter_nodes():
            sbu = self.sbus.get(node.name)
            if not sbu == None:
                if node.name[0] == '_':
                    # Edges need to have a supplementary translation if the neighbors are not equally large
                    neighbor0, neighbor1 = node.neighbors
                    radii = []
                    for neighbor in node.neighbors:
                        neighbor_prefix, neighbor_wyckoff_name, neighbor_number, neighbor_suffix = split_name(neighbor.name)
                        unit_neighbor = self.topology.get_node(neighbor_prefix + neighbor_wyckoff_name + str(neighbor_number))
                        radii.append(self.sbus[unit_neighbor.name].get_radius())
                    r0, r1 = radii
                    r_edge = sbu.get_radius()
                    k = 0.5*(r0-r1)/(r0 + 2*r_edge + r1)
                    edge_cart_pos = self.topology.unit_cell.to_cartesian(node.frac_pos) + k*(self.topology.unit_cell.to_cartesian(node.neighbors[1].frac_pos) - self.topology.unit_cell.to_cartesian(node.neighbors[0].frac_pos))
                    node.frac_pos = self.topology.unit_cell.to_fractional(edge_cart_pos)
                sbu.translate(self.topology.unit_cell.to_cartesian(node.frac_pos) - sbu.center)

    def _init_configurations(self):
        '''
        3) Get all possible neighbor combinations with the smallest RMSD

        For every neighbor combination, compute the RMSD. All combinations with
        the smallest RMSD (with an acceptance of delta) are retained. Remark that
        the number of screened combinations is N!, with N the coordination number
        of the node.

        TO DO: Find a shortcut to reduce the number of combinations if N becomes
        larger than 8. Idea: use symmetry
        '''
        for node_name, sbu in self.sbus.items():
            node = self.topology.get_node(node_name)
            for combination in permutations(node.neighbors):
                self.configurations[node_name].append(combination)
                self.configurations_data[node_name][combination] = [float('nan'), float('nan'), float('nan')]

    def reduce_configurations_geometric(self, node, delta = 0.005*angstrom):
        sbu = self.sbus[node.name]
        poe_pos = [normalize(poe[0] - sbu.center) for poe in sbu.poes]
        result_configurations = []
        min_rmsd = None
        for configuration in self.configurations[node.name]:
            nei_pos = [normalize(self.topology.unit_cell.to_cartesian(neighbor.frac_pos) - self.topology.unit_cell.to_cartesian(node.frac_pos)) for neighbor in configuration]
            rmsd = sbu.kabsch(np.array(nei_pos), np.array(poe_pos), apply = False, only_rotation = True)
            self.configurations_data[node.name][configuration][0] = rmsd
            if not min_rmsd == None and abs(min_rmsd - rmsd) < delta:
                # Degeneration
                result_configurations.append(configuration)
            elif min_rmsd == None or rmsd < min_rmsd:
                min_rmsd = rmsd
                result_configurations = [configuration]
        self.configurations[node.name] = result_configurations
        self.rmsd[node.name] = min_rmsd

    def reduce_configurations_covalent(self, node, delta = 25*kjmol):
        sbu = self.sbus[node.name]
        result_configurations = []
        min_energy = None
        cov_energies = []
        configurations_name = []
        for configuration in self.configurations[node.name]:
            sbu_conf_name = '-'.join([node.name] + [nei.name for nei in configuration])
            cov_energy = self.test_configuration(node, configuration, cov = True)
            self.configurations_data[node.name][configuration][1] = cov_energy
            if not min_energy == None and abs(min_energy - cov_energy) < delta:
                # Degeneration
                result_configurations.append(configuration)
            elif min_energy == None or cov_energy < min_energy:
                min_energy = cov_energy
                result_configurations = [configuration]
        self.configurations[node.name] = result_configurations

    def set_final_configuration(self, node):
        from random import choice
        final_configuration = choice(self.configurations[node.name])
        self.final_configurations[node.name] = final_configuration

    def get_random_structure(self):
        for node_name, sbu in self.sbus.items():
            node = self.topology.get_node(node_name)
            self.reduce_configurations_geometric(node)
            self.set_final_configuration(node)
            self.test_configuration(node, self.final_configurations[node.name], cov = True, apply = True)

    def get_nucleation_structure(self):
        '''
        Nucleation build:
            4a) Iterate breadth first through the Topology Graph
            4b) For every node encountered, choose the neighbor combination that minimizes the covalent energy
            4c) Add the SBU in the chosen neighbor combination to the Cluster
        '''
        graph = self.topology.create_graph()
        for index, dist in graph.iter_breadth_first():
            node_name = graph.node_names[index]
            node = self.topology.get_node(node_name)
            sbu = self.sbus.get(node_name)
            if not sbu == None:
                self.reduce_configurations_geometric(node)
                self.reduce_configurations_covalent(node, delta = 0.0*kjmol)
                self.set_final_configuration(node)
                self.test_configuration(node, self.final_configurations[node.name], cov = True, apply = True)


    def test_configuration(self, node, configuration, cov = False, apply = False, N = 36):
        '''
        Get the energy if a combination for a certain node is added to the Cluster.
        If apply is False [default], the SBU is afterwards removed from the Cluster,
        if True it stays in the Cluster.

        If the node is an edge and do_scan is True, a rotational scan is performed
        to fix the rotational degree of freedom of the linker. N defines the number
        of points screened during this scan.
        '''
        # Prepare SBU and add to cluster
        sbu = self.sbus[node.name]
        nei_pos = []
        neighbor_names_with_sbu = []
        node_pos = self.topology.unit_cell.to_cartesian(node.frac_pos)
        for poe, neighbor in zip(sbu.poes, configuration):
            neighbor_sbu = self.get_neighbor_sbu(node.name, neighbor.name)
            poe[2] = neighbor.name
            if neighbor_sbu in self.cluster.sbus:
                neighbor_names_with_sbu.append(neighbor.name)
            neighbor_pos = self.topology.unit_cell.to_cartesian(neighbor.frac_pos)
            nei_pos.append(normalize(neighbor_pos - node_pos))
        sbu.orient(nei_pos, only_rotation = True)
        self.cluster.add_sbu(sbu)
        sbu_index = len(self.cluster.sbus) - 1
        for neighbor in configuration:
            neighbor_sbu = self.get_neighbor_sbu(node.name, neighbor.name)
            if neighbor_sbu in self.cluster.sbus:
                self.add_bond(node.name, neighbor.name)
        
        # Compute energy of configuration
        if len(sbu.poes) == 2:
            energy = self.linker_rotation(node, sbu, configuration, sbu_index, neighbor_names_with_sbu, cov = cov, N = N)
        else:
            if cov:
                energy = 0.0
                for neighbor_name in neighbor_names_with_sbu:
                    energy += self.get_bond_energy(node.name, neighbor_name)
            else:
                ff = self.cluster.get_ff()
                energy = ff.compute()
        
        
        # If the configuration should not be applied, the sbu is removed again
        if apply:
            self.final_configurations[node.name] = configuration
        else:
            self.cluster.remove_sbu(sbu)
        return energy

    def linker_rotation(self, node, sbu, configuration, sbu_index, neighbor_names_with_sbu, cov = False, N = 36):
        '''
        For a linker, with two points of extension, there is still a degree of
        freedom once these are oriented towards the neighboring nodes. This degree
        of freedom is fixed here by performing a rotational scan after which the
        configuration with the smallest covalent energy is chosen.

        The covalent energy is chosen as this is independent of the configuration
        of other linkers and it already gives a good indication of the optimal configuration.
        After the structure is build, the other interactions can be included by
        performing an optimization.
        '''
        # Linker: a rotation scan has to be applied
        axis = normalize(sbu.poes[1][0] - sbu.poes[0][0])
        alpha = 2*np.pi/N
        min_energy = None
        min_alpha = None
        angles = []
        energies = []
        systems = []
        for i in range(N):
            sbu.rotate_about_center(alpha, axis)
            if cov:
                energy = 0.0
                for neighbor_name in neighbor_names_with_sbu:
                    energy += self.get_bond_energy(node.name, neighbor_name)
            else:
                ff = self.cluster.get_ff()
                energy = ff.compute()
            angles.append((i+1)*alpha)
            energies.append(energy)
            if min_energy == None or energy < min_energy:
                min_energy = energy
                min_angle = (i+1)*alpha
        sbu.rotate_about_center(min_angle, axis)
        for neighbor_name in neighbor_names_with_sbu:
            self.get_bond_energy(node.name, neighbor_name) # Update FFs
        return min_energy


    def add_bond(self, vertex0, vertex1):
        sbu0 = self.sbus[vertex0]
        sbu1 = self.get_neighbor_sbu(vertex0, vertex1)
        atom_index0, atom_index1 = self.get_atom_bond_indices(vertex0, vertex1)
        # Add bonds
        if [atom_index0, atom_index1] in self.cluster.atom_bonds or [atom_index1, atom_index0] in self.cluster.atom_bonds: return
        self.cluster.atom_bonds.append([atom_index0, atom_index1])
        # Add parameters
        pair = [sbu0.name + sbu0.termination, sbu1.name + sbu1.termination]
        if self.cluster.combine_parameters and not (pair in self.cluster.ff_bonds or pair[::-1] in self.cluster.ff_bonds):
            self.cluster.ff_bonds.append(pair)
            self.cluster.parameters.add_combination(sbu0.parameters, sbu1.parameters)
            self.cluster.uff_parameters.add_combination(sbu0.uff_parameters, sbu1.uff_parameters)
        self.cluster.ff = None

    def get_atom_bond_indices(self, vertex0, vertex1):
        vertex0_name = self.topology.get_unit_node(vertex0).name
        sbu0 = self.sbus[vertex0_name]
        sbu1 = self.get_neighbor_sbu(vertex0_name, vertex1)
        sbus = [sbu0, sbu1]
        vertices = [vertex0, vertex1]
        # Check if poes are filled
        for sbu in [sbu0, sbu1]:
            no_poe_connection = sum([poe[2] == None for poe in sbu.poes])
            if not no_poe_connection == 0:
                print('WARNING: SBU {} has only {} out of {} points of connection that are assigned to a neighbor'.format(sbu.name, len(sbu.poes) - no_poe_connection, len(sbu.poes)))
        # Find poes that are connected to each other
        poes = []
        prefix1, wyckoff_name1, index1, suffix1 = split_name(vertices[1])
        if self.sbus.get(prefix1 + wyckoff_name1 + str(index1)) == None:
            # Vertex1 is edge that has no SBU assigned
            # Both SBUs should have a poe that is pointed towards this edge
            for poe0 in sbus[0].poes:
                for poe1 in sbus[1].poes:
                    poe0_prefix, poe0_wyckoff_name, poe0_index, poe0_suffix = split_name(poe0[2])
                    poe1_prefix, poe1_wyckoff_name, poe1_index, poe1_suffix = split_name(poe1[2])
                    poe0_unit_name = poe0_prefix + poe0_wyckoff_name + str(poe0_index)
                    poe1_unit_name = poe1_prefix + poe1_wyckoff_name + str(poe1_index)
                    if poe0_unit_name == poe1_unit_name and poe0_unit_name == prefix1 + wyckoff_name1 + str(index1):
                        if np.linalg.norm(poe0[0] - poe1[0]) < 1e-3 and poe0[1] == poe1[1] and poe0[2] == poe1[2]: continue
                        if len(poes) == 0:
                            poes = [poe0, poe1]
                        else:
                            flag0 = False
                            flag1 = False
                            for poe in poes:
                                if np.linalg.norm(poe[0] - poe0[0]) < 1e-3 and poe[1] == poe0[1] and poe[2] == poe0[2]:
                                    flag0 = True
                                if np.linalg.norm(poe[0] - poe1[0]) < 1e-3 and poe[1] == poe1[1] and poe[2] == poe1[2]:
                                    flag1 = True
                            assert flag0 and flag1
        else:
            for i, j in [[0, 1], [1, 0]]:
                prefix0, wyckoff_name0, index0, suffix0 = split_name(vertices[i])
                prefix1, wyckoff_name1, index1, suffix1 = split_name(vertices[j])
                for poe in sbus[j].poes:
                    poe_prefix, poe_wyckoff_name, poe_index, poe_suffix = split_name(poe[2])
                    if poe_prefix == prefix0 and poe_wyckoff_name == wyckoff_name0 and poe_index == index0 and np.linalg.norm(get_diff(poe_suffix) + get_diff(suffix1) - get_diff(suffix0)) < 1e-3:
                        poes.append(poe)

        if not len(poes) == 2:
            raise RuntimeError('In SBUs {} and {}, {} points of extension that have to be connected are found, instead of 2'.format(sbu0.name, sbu1.name, len(poes)))
        return poes[0][1], poes[1][1]

    def get_bond_energy(self, vertex0, vertex1):
        atom_bond = self.get_atom_bond_indices(vertex0, vertex1)
        return self.cluster.get_bond_energy(atom_bond)

    def get_neighbor_sbu(self, node_name, neighbor_name):
        '''
        Returns the SBU neighboring to the node with the name node_name in the
        direction of the neighbor with name neighbor_name (which is an edge),
        irrespective from if their is put an SBU on this edge or not.
        Node_name has to be inside the unit cell (no suffix)
        Neighbor_name can be outside the unit cell
        '''
        unit_neighbor = self.topology.get_unit_node(neighbor_name)
        neighbor_sbu = self.sbus.get(unit_neighbor.name)
        if neighbor_sbu == None:
            # Edge with no sbu assigned
            if not (unit_neighbor.name[0] == '_' and unit_neighbor.cn == 2):
                raise RuntimeError('The vertex {} has no SBU'.format(unit_neighbor.name))
            for neighbor_neighbor in unit_neighbor.neighbors:
                unit_neighbor_neighbor = self.topology.get_unit_node(neighbor_neighbor.name)
                if not unit_neighbor_neighbor.name == node_name:
                    neighbor_sbu = self.sbus.get(unit_neighbor_neighbor.name)
        if neighbor_sbu == None:
            # Edge with no sbu assigned connected to the same vertices
            # e.g.: sql
            for neighbor_neighbor in unit_neighbor.neighbors:
                nn_prefix, nn_wyckoff, nn_index, nn_suffix = split_name(neighbor_neighbor.name)
                assert nn_prefix + nn_wyckoff + str(nn_index) == node_name
            neighbor_sbu = self.sbus.get(node_name)
        return neighbor_sbu

    def output(self, folder):
        if folder[-1] == '/':
            name = folder.split('/')[-2]
        else:
            name = folder.split('/')[-1]
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.cluster.to_chk(os.path.join(folder, name + '.chk'))
        fn_cluster = os.path.join(folder, 'pars_cluster.txt')
        fn_uff = os.path.join(folder, 'pars_uff.txt')
        self.cluster.parameters.to_file(fn = fn_cluster)
        self.cluster.uff_parameters.to_file(fn = fn_uff)

def normalize(v):
    return v/np.linalg.norm(v)

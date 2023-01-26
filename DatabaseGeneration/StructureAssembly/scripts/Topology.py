#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from copy import deepcopy

from molmod.unit_cells import UnitCell
from molmod.units import deg, angstrom
from molmod.graphs import Graph
import numpy as np

class Node():
    '''
    A node in a topology, this can either be an edge (coordination number cn = 2) or a vertex (cn > 2)

    Attributes:
        name [string]: the name of the Node, consisting of a prefix, a letter combination, a number and eventually a suffix
            The prefix is '_' if the Node is an edge and is '' (no prefix) if the Node is a vertex
            The letter combination designs the WyckoffSet where the Node is part from
            The number is the index of the Node in the WyckoffSet
            The suffix is a string of three characters (each being '-', '0' or '+') which mentions if the node is outside the unit cell
            Examples:
                X3-00 is the third vertex in the Wyckoff vertex Set A, translated over one time the negative first unit cell vector
                _AF12 is the twelfth edge in the Wyckoff edge Set _AF (as the edge is in the unit cell, the suffix is not present)
        cn [integer]: coordination number of the Node, i.e. the number of neighbors the Node has
        frac_pos [numpy array]: an array with the fractional coordinates of the Node with respect to the unit cell of the Topology
        neighbors [array of Nodes]: neighboring Nodes of this Node

    TO DO: make of the prefix, wyckoff_name, number, and suffix also an argument??
    '''
    def __init__(self, name, cn, frac_pos, neighbors = None):
        self.name = name
        self.cn = cn
        self.frac_pos = frac_pos
        if neighbors == None:
            neighbors = []
        self.neighbors = neighbors

    def __str__(self):
        if self.cn == 2:
            type = 'EDGE'
        else:
            type = 'VERTEX'
        return '{} {} ({}) - {} (Neighbors: {})'.format(type, self.name, self.cn, self.frac_pos, ', '.join([neighbor.name for neighbor in self.neighbors]))

    def add_neighbor(self, neighbor):
        '''
        Add a neighbor to the Node
        '''
        if len(self.neighbors) >= self.cn:
            print('WARNING: Node {} already has {} neighbors, while its coordination number is {}, you will have to remove neighbors in the future'.format(self.name, len(self.neighbors), self.cn))
        self.neighbors.append(neighbor)

    def copy(self, suffix = None):
        '''
        Returns an independent copy of the Node, if suffix is given, a Node outside the unit cell is given.
        The suffix is a string of three characters. The ith character is '-', '0' or '+' and determines if the Node is translated over the ith unit cell vector or not
        '''
        if suffix == None or suffix == '000':
            suffix = '000'
            new_name = self.name
        else:
            new_name = self.name + suffix
        new_neighbors = []
        for neighbor in self.neighbors:
            new_neighbors.append(neighbor)
        return Node(new_name, self.cn, np.copy(self.frac_pos) + get_diff(suffix), neighbors = new_neighbors)

class WyckoffSet():
    '''
    A WyckoffSet is a set of Nodes in the topology that are connected by symmetry.

    Attributes:
        name [string]: the name of the WyckoffSet, which has the same construction as the name of a Node, but without the number
            Examples:
                X3-00 is the third vertex in the Wyckoff vertex Set A, translated over one time the negative first unit cell vector
                _AF12 is the twelfth edge in the Wyckoff edge Set _AF (as the edge is in the unit cell, the suffix is not present)
        nodes [list of Nodes]: some nodes that are present in the WyckoffSet. Nodes can also be added to the WyckoffSet afterwards.
        cn [int]: The coordination number of all nodes in the WyckoffSet, this is used to check if a Node can be allowed in the WyckoffSet
    '''
    def __init__(self, name, nodes = None):
        self.name = name
        if nodes == None:
            nodes = []
        self.nodes = nodes
        self.cn = None
        self.check_nodes(nodes)

    def check_nodes(self, nodes):
        '''
        Checks that all the nodes in the WyckoffSet have the same coordination number
        '''
        if isinstance(nodes, Node):
            nodes = [nodes]
        if self.cn == None:
            if len(nodes) > 0:
                self.cn = nodes[0].cn
        for node in nodes:
            if not self.cn == node.cn:
                raise RuntimeError('The nodes in the Wyckoff set do not have the same coordination number')

    def __str__(self):
        if self.cn == 2:
            type = 'EDGE'
        else:
            type = 'VERTEX'
        text = 'WYCKOFF {} {} ({} nodes)\n'.format(type, self.name, len(self))
        for node in self.nodes:
            text += '{}\n'.format(node)
        return text

    def __len__(self):
        return len(self.nodes)

    def iter_nodes(self):
        '''
        Iterate over the nodes of the WyckoffSet
        '''
        for node in self.nodes:
            yield node

    def add_node(self, node):
        '''
        Add a new node to the WyckoffSet
        '''
        self.check_nodes(node)
        self.nodes.append(node)

    def copy(self):
        new_name = self.name
        new_nodes = []
        for node in self.nodes:
            new_nodes.append(node.copy())
        return WyckoffSet(new_name, new_nodes)

class Topology():
    '''
    A Topology is a periodic graph, which represents the interconnectivity between nodes.

    Arguments
        name [string]: a three-letter sequence given by the RCSR, potentialy followed by one or more extensions
            REMARK: the extension '*' is replaced by a '+' in order to avoid command line problems
        unit cell [molmod UnitCell]: a unit cell to indicate the periodicity of the topology
        wyckoff_vertices [dict of WyckoffSets]: the WyckoffSets of vertices that are present in the topology
        wyckoff_edges [dict of WyckoffSets]: the WyckoffSets of edges that are present in the topology
        nvertices [int]: number of vertices in the Topology
        nedges [int]: number of edges in the Topology
    '''
    
    topology_path = '../data/Topologies/'

    def __init__(self, name, unit_cell, wyckoff_vertices = None, wyckoff_edges = None, dimension = None):
        self.name = name
        self.unit_cell = unit_cell
        if wyckoff_vertices == None:
            wyckoff_vertices = {}
        if wyckoff_edges == None:
            wyckoff_edges = {}
        self.wyckoff_vertices = wyckoff_vertices
        self.wyckoff_edges = wyckoff_edges
        self.nvertices = sum([len(wyckoff_vertex) for wyckoff_vertex in self.wyckoff_vertices.values()])
        self.nedges = sum([len(wyckoff_edge) for wyckoff_edge in self.wyckoff_edges.values()])
        self.dimension = dimension

    def copy(self):
        new_unit_cell = self.unit_cell.copy_with()
        new_wyckoff_vertices = {}
        for name, wyckoff_vertex in self.wyckoff_vertices.items():
            new_wyckoff_vertices[name] = wyckoff_vertex.copy()
        new_wyckoff_edges = {}
        for name, wyckoff_edge in self.wyckoff_edges.items():
            new_wyckoff_edges[name] = wyckoff_edge.copy()
        return Topology(self.name, new_unit_cell, new_wyckoff_vertices, new_wyckoff_edges, self.dimension)

    @classmethod
    def load(cls, top_name, dimension = None):
        if dimension is None:
            if '{}.top' in os.listdir(cls.topology_path + '2D'):
                dimension = '2D'
            elif '{}.top' in os.listdir(cls.topology_path + '3D'):
                dimension = '3D'
            else:
                raise IOError('Could not find topology file for topology ' + top_name)
        return cls.from_file(cls.topology_path + dimension + '/' + top_name + '.top', dimension)

    @classmethod
    def from_file(cls, fn, dimension):
        '''
        Load the topology from a .top file.

        A .top file consists of 4 sections, each preceded by an information line indicating the format (except for the first section):

        Section 1: general information (name of topology + number of vertices + number of edges) (1 line)
        Section 2: unit cell information (length of the unit cell vectors and the angles between them) (1 line)
        Section 3: Information on the vertices (1 line per vertex)
            Each line consists of the following information:
                Name of the vertex
                Coordination number of the vertex
                Fractional position of the vertex
                Point group of the vertex
        Section 4: Information on the edges (1 line per edge)
            Each line consists of the following data
                Name of the edge
                Name of the vertices that it connects

        An example of such a .top file (hcb.top):

        hcb 2 3
        # a     b       c       alpha   beta    gamma
        1.7321  1.7321  10.0    90.0    90.0    120.0
        # V     cn      frac_x  frac_y  frac_z  PG
        A1    3       0.3333  0.6667  0.0     1
        A2    3       0.6667  0.3333  0.0     1
        # E     N1      N2
        _A1   A1      A2-00
        _A2   A1      A2
        _A3   A2      A10-0

        Note that the information line has to be present and no empty lines are allowed.
        TO DO: Allow more flexibility in .top files
        '''
        if not fn.endswith('.top'):
            if '.' in fn:
                raise RuntimeError('Other extension than .top recognized in filename {}'.format(fn))
            else:
                fn = fn + '.top'
        with open(fn, 'r') as f:
            # Section 1: General information
            name, nvertices, nedges = f.readline().rstrip().split()
            nvertices, nedges = [int(element) for element in [nvertices, nedges]]
            # Section 2: Unit cell information
            if not f.readline().rstrip() == '# a\tb\tc\talpha\tbeta\tgamma':
                raise RuntimeError('Error while reading file {}: unit cell line (2nd line) not as expected'.format(fn))
            a, b, c, alpha, beta, gamma = [float(element) for element in f.readline().rstrip().split()]
            unit_cell = UnitCell.from_parameters3([a, b, c], [alpha*deg, beta*deg, gamma*deg])

            # Create empty topology, nodes have to be added
            result = cls(name, unit_cell, dimension = dimension)

            # Section 3: vertex information
            if not f.readline().rstrip() == '# V\tcn\tfrac_x\tfrac_y\tfrac_z\tPG':
                raise RuntimeError('Error while reading file {}: vertex line (4th line) not as expected'.format(fn))
            wyckoff_vertices = {}
            for i in range(nvertices):
                vertex_name, cn, frac_x, frac_y, frac_z, pg = f.readline().rstrip().split()
                cn = int(cn)
                frac_pos = np.array(list(map(float, [frac_x, frac_y, frac_z])))
                prefix, wyckoff_name, number, suffix = split_name(vertex_name)
                vertex = Node(prefix + wyckoff_name + str(number), cn, frac_pos)
                wyckoff_vertex = wyckoff_vertices.setdefault(prefix + wyckoff_name, WyckoffSet(prefix + wyckoff_name))
                wyckoff_vertex.add_node(vertex)
            result.wyckoff_vertices = wyckoff_vertices
            result.nvertices = nvertices

            # Section 4: edge information
            if not f.readline().strip() == '# E\tN1\tN2':
                raise RuntimeError('Error while reading file {}: edge line (last comment line) not as expected'.format(fn))
            wyckoff_edges = {}
            bonds = []
            for i in range(nedges):
                edge_name, n1, n2 = f.readline().rstrip().split()
                prefix, wyckoff_name, number, suffix = split_name(edge_name)
                neighbor1 = result.get_node(n1)
                neighbor2 = result.get_node(n2)
                frac_pos = (neighbor1.frac_pos + neighbor2.frac_pos)/2
                edge = Node(prefix + wyckoff_name + str(number), 2, frac_pos, neighbors = [neighbor1, neighbor2])
                for neighbor in [neighbor1, neighbor2]:
                    # Add the edge to the neighbors of both neighbors
                    neighbor_prefix, neighbor_wyckoff_name, neighbor_number, neighbor_suffix = split_name(neighbor.name)
                    if neighbor_suffix == '000':
                        unit_neighbor = neighbor
                    else:
                        unit_neighbor = result.get_node(neighbor_prefix + neighbor_wyckoff_name + str(neighbor_number))
                    unit_neighbor = result.get_unit_node(neighbor.name)
                    suffix_difference = get_suffix_difference(edge, neighbor)
                    if suffix_difference == '000':
                        unit_neighbor.add_neighbor(edge)
                    else:
                        unit_neighbor.add_neighbor(edge.copy(suffix_difference))
                wyckoff_edge = wyckoff_edges.setdefault(prefix + wyckoff_name, WyckoffSet(prefix + wyckoff_name))
                wyckoff_edge.add_node(edge)
            result.wyckoff_edges = wyckoff_edges
            result.nedges = nedges
        result.check()
        return result

    def check(self):
        '''
        Check that all nodes have the same number of neighbors as the coordination number indicates
        '''
        for node in self.iter_nodes():
            if not node.cn == len(node.neighbors):
                raise RuntimeError('Node {} has not the same numbers of neighbors {} as the coordination number {} indicates'.format(node.name, len(node.neighbors), node.cn))

    def __str__(self):
        text = 'TOPOLOGY {} ({} vertices, {} edges)\n\n'.format(self.name, self.nvertices, self.nedges)
        for col, name in enumerate(['a', 'b', 'c']):
            text += '{} - {}\n'.format(name, np.round(self.unit_cell.matrix[:, col], 4))
        text += '\n'
        for wyckoff_vertex in self.wyckoff_vertices.values():
            text += '{}\n'.format(wyckoff_vertex)
        for wyckoff_edge in self.wyckoff_edges.values():
            text += '{}\n'.format(wyckoff_edge)
        return text

    def iter_vertices(self):
        for wyckoff_name in sorted(self.wyckoff_vertices.keys()):
            for vertex in self.wyckoff_vertices[wyckoff_name].iter_nodes():
                yield vertex

    def iter_edges(self):
        for wyckoff_name in sorted(self.wyckoff_edges.keys()):
            for edge in self.wyckoff_edges[wyckoff_name].iter_nodes():
                yield edge

    def iter_nodes(self):
        for vertex in self.iter_vertices():
            yield vertex
        for edge in self.iter_edges():
            yield edge

    def get_node(self, node_name):
        '''
        Returns the node with node_name
        If the node is in the unit cell (suffix '000'), the node is returned, if not, a copy of the node is returned
        '''
        prefix, wyckoff_name, number, suffix = split_name(node_name)
        diff = get_diff(suffix)
        number = number - 1
        if prefix == '':
            node = self.wyckoff_vertices[wyckoff_name].nodes[number]
        else:
            node = self.wyckoff_edges[prefix + wyckoff_name].nodes[number]
        if not suffix == '000':
            node = node.copy(suffix)
        return node

    def get_unit_node(self, node_name):
        '''
        The (equivalent) node in the unit cell is returned
        '''
        prefix, wyckoff_name, number, suffix = split_name(node_name)
        return self.get_node(prefix + wyckoff_name + str(number))

    def rescale(self, factors):
        '''
        Rescale the unit cell of the topology
        Either factors is the rescaling factor of an isotropic rescaling,
        or it is a list containing the rescaling factors of te x-, y-, and z-axis for an anisotropic rescaling
        '''
        if isinstance(factors, float):
            factors = [factors]*3
        lengths, angles = self.unit_cell.parameters
        for i in range(3):
            lengths[i] *= factors[i]
        if self.dimension == '2D':
            lengths[2] = 10*angstrom
        self.unit_cell = UnitCell.from_parameters3(lengths, angles)

    def create_graph(self):
        '''
        Construct a graph from the topology
        TO DO: Make of the Topology a Graph object and give the nodes to a dictionary??
        '''
        graph_vertices = []
        graph_edges = []
        for node in self.iter_nodes():
            graph_vertices.append(node)
            for neighbor in node.neighbors:
                unit_neighbor = self.get_unit_node(neighbor.name)
                if unit_neighbor in graph_vertices:
                    index0 = graph_vertices.index(node)
                    index1 = graph_vertices.index(unit_neighbor)
                    graph_edges.append([index0, index1])
        graph = MyGraph(graph_edges)
        graph.node_names = dict((i, graph_vertices[i].name) for i in range(len(graph_vertices)))
        return graph

def split_name(node_name):
    '''
    Splits the name of each node in the prefix, the letter of the WyckoffSet, the number of the Node in the WyckoffSet and the suffix
    Returns result = [prefix, name, number, suffix]
        prefix = '' in case of vertex, '_' in case of edge
        name = letters (like 'A', 'AB', ...) that indicate the WyckoffSet
        number = 12 (integer)
        suffix = three character string to indicate the position relative to the unit cell
    '''
    result = []
    if '-' in node_name or '+' in node_name or 'p' in node_name or 'm' in node_name:
        suffix = node_name[-3:]
        node_name = node_name[:-3]
    else:
        suffix = '000'
        if node_name[-3:] == '000':
            raise RuntimeError('Not sure what to do with node {}, it seems like the suffix 000 is already present'.format(node_name))
    if node_name[0] == '_':
        result.append('_')
        node_name = node_name[1:]
    else:
        result.append('')
    if node_name[1].isdigit():
        result.append(node_name[0])
        result.append(int(node_name[1:]))
    else:
        if not node_name[2].isdigit():
            raise RuntimeError('Vertex name {} is not recognized, should be the name of the Wyckoff set (max. 2 letters) followed by a number'.format(vertex_name))
        result.append(node_name[:2])
        result.append(int(node_name[2:]))
    result.append(suffix)
    return result

def wyckoff_name_to_number(wyckoff_name):
    '''
    Returns the index of the WyckoffSet: 'A' -> 1, ..., 'Z' -> 26, 'AA' -> 27, ...
    '''
    if wyckoff_name[0] == '_':
        wyckoff_name = wyckoff_name[1:]
    if len(wyckoff_name) == 1:
        return ord(wyckoff_name[0]) - 65
    elif len(wyckoff_name) == 2:
        return (ord(wyckoff_name[0]) - 65)*26 + (ord(wyckoff_name[1]) - 65)
    else:
        raise NotImplementedError('Wyckoff name {} has to many characters'.format(wyckoff_name))

def wyckoff_number_to_name(wyckoff_number, edge = False):
    i = wyckoff_number
    if i < 26:
        name = chr(i + 65)
    else:
        j = -1
        while i >= 26:
            j += 1
            i -= 26
        name = chr(j + 65) + chr(i + 65)
    if edge:
        name = '_' + name
    return name

def get_diff(suffix):
    '''
    Get the fractional difference vector associated with the suffix
    '''
    diff = np.array([0., 0., 0.])
    for i, char in enumerate(suffix):
        if char == 'p':
            diff[i] += 2.0
        elif char == '+':
            diff[i] += 1.0
        elif char == '-':
            diff[i] -= 1.0
        elif char == 'm':
            diff[i] -= 2.0
        else:
            if not char == '0':
                raise RuntimeError('Suffix {} not recognized'.format(suffix))
    return diff

def get_suffix(diff):
    '''
    Get the suffix associated with the fractional difference vector
    '''
    suffix = ''
    for i in diff:
        if i == -2:
            suffix += 'm'
        elif i == -1:
            suffix += '-'
        elif i == +1:
            suffix += '+'
        elif i == +2:
            suffix += 'p'
        else:
            if not i == 0:
                raise RuntimeError('Difference {} not recognized'.format(diff))
            suffix += '0'
    return suffix

def get_suffix_difference(node1, node2):
    '''
    Returns the suffix that indicates the difference between the suffixes of node2 and node1
    Example:
        node1 = A100+
        node2 = B1-00
        result = suffix2 - suffix1 = 00+ - -00 = +0+
    '''
    diffs = []
    for node in [node1, node2]:
        prefix, wyckoff_name, number, suffix = split_name(node.name)
        diff = get_diff(suffix)
        diffs.append(diff)
    final_diff = diffs[0] - diffs[1]
    return get_suffix(final_diff)

class MyGraph(Graph):
    def iter_breadth_first(self, start=None, do_paths=False, do_duplicates=False):
        """Iterate over the vertices with the breadth first algorithm.
           See http://en.wikipedia.org/wiki/Breadth-first_search for more info.
           If not start vertex is given, the central vertex is taken.
           By default, the distance to the starting vertex is also computed. If
           the path to the starting vertex should be computed instead, set path
           to True.
           When duplicate is True, then vertices that can be reached through
           different  paths of equal length, will be iterated twice. This
           typically only makes sense when path==True.
        """
        from collections import deque
        work = np.zeros(self.num_vertices, int)
        work[:] = -1
        while -1 in work:
            if start is None:
                sub_graph = self.get_subgraph([i for i in range(self.num_vertices) if work[i] == -1])
                start_ind = sub_graph.central_vertex
            else:
                try:
                    start_ind = int(start)
                except ValueError:
                    raise TypeError("First argument (start) must be an integer.")
                if start_ind < 0 or start_ind >= self.num_vertices:
                    raise ValueError("start must be in the range [0, %i[" %
                                     self.num_vertices)
            work[start_ind] = 0
            if do_paths:
                result = (start_ind, 0, (start_ind, ))
            else:
                result = (start_ind, 0)
            yield result
            todo = deque([result])
            while len(todo) > 0:
                if do_paths:
                    parent, parent_length, parent_path = todo.popleft()
                else:
                    parent, parent_length = todo.popleft()
                current_length = parent_length + 1
                for current in self.neighbors[parent]:
                    visited = work[current]
                    if visited == -1 or (do_duplicates and visited == current_length):
                        work[current] = current_length
                        if do_paths:
                            current_path = parent_path + (current, )
                            result = (current, current_length, current_path)
                        else:
                            result = (current, current_length)
                        yield result
                        todo.append(result)

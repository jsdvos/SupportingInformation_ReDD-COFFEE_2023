import os
import sys

import numpy as np

from yaff import System, log
log.set_level(0)
from molmod import MolecularGraph, GraphSearch, CustomPattern, GraphError
from molmod.periodic import periodic as pt
from molmod.units import angstrom

from utils import patterns, cc, cn, crit1, crit2, endict, poldict

def get_isolated_parts(graph):
    work = [-1]*graph.num_vertices
    n = 0
    while -1 in work:
        start = work.index(-1)
        for i, d in graph.iter_breadth_first(start):
            work[i] = n
        n += 1
    return work

def clean_system(system, threshold = 12):
    '''
    threshold: if an isolated part has less than this number of atoms, it considered
    as an isolated molecule and removed. If it is larger than this, it is considered
    as being part of the framework (e.g. 2D layers or 3D-catenated nets are not connected).
        Default = 12: would allow a hcb system with a 3c-triazine ring and no linkers
        Also checked on CURATED database and 2D layers always have at least 15 atoms
    A more decent approach to clean the system is implemented in the preprocessing routine
    '''
    graph = MolecularGraph(system.bonds, system.numbers)
    parts = get_isolated_parts(graph)
    mask = [True]*system.natom
    n_final = max(parts) + 1
    for i in range(n_final):
        n = sum([k == i for k in parts])
        if n < threshold:
            n_final -= 1
            for k in range(len(parts)):
                if i == parts[k]:
                    mask[k] = False
    indices = np.array([i for i in range(system.natom)])[mask]
    system = system.subsystem(indices)
    graph = graph.get_subgraph(indices, normalize = True)
    return system, graph, n_final

def get_ligands(system, fn_out, n_parts):
    # Define molecular graph
    # In small systems, a supercell is taken to make sure that no
    # artificial isolated parts would be created. This would happen
    # when linkers are connected to the same building block in different
    # unit cells, which can only happen if the unit cell in that direction
    # is small. Here 50A is taken as threshold for a cell length.
    reps = [1, 1, 1]
    for i, cell_length in enumerate(system.cell.parameters[0]):
        if cell_length < 50*angstrom:
            reps[i] = 2
    system = system.supercell(*reps)
    graph = MolecularGraph(system.bonds, system.numbers)
    n_parts = max(get_isolated_parts(graph)) + 1
    # Get atom indices of ligands
    indices = set([])
    for ligand in sorted(patterns.keys(), key=lambda e: len(patterns[e][1]), reverse = True):
        pattern, allowed = patterns[ligand]
        ligand_indices = set([])
        # Search for pattern
        graph_search = GraphSearch(pattern)
        new_graph = graph
        while True:
            for match in graph_search(new_graph):
                if ligand in crit1.keys():
                    # Extra criterium 1: exact neighbors
                    # e.g.: pyridine ring with propyl and methyl functionalized
                    # nitrogen and carbon-2 atoms would be detected as imine bond
                    nneighs = crit1[ligand]
                    correct_neighs = []
                    c_sp3_hh = False
                    for i in range(len(nneighs[0])):
                        correct_neighs.append(len(graph.neighbors[match.forward[i]]))
                        if correct_neighs[-1] == 4 and not (ligand == 'Amine' and i == 0):
                            # If there are 4 neighbors, it is meant to allow for adamantane
                            # Errors are obtained when an sp3 carbon with two hydrogens is present
                            # E.g. linker 41 in Martin2014
                            count_h = 0
                            for j in graph.neighbors[match.forward[i]]:
                                if system.numbers[j] == 1:
                                    count_h += 1
                            c_sp3_hh = (count_h > 1)
                    if not correct_neighs in nneighs or c_sp3_hh:
                        continue
                if ligand in crit2.keys():
                    # Extra criterium 2: end carbons are not allowed to connect
                    # This avoids detecting a pyridine as an imine bond
                    not_connected = crit2[ligand]
                    not_connected = [match.forward[i] for i in not_connected]
                    if len(not_connected) == 0:
                        # Check all neighbors of the pattern
                        for i in range(pattern.pattern_graph.num_vertices):
                            if i not in allowed:
                                for j in graph.neighbors[match.forward[i]]:
                                    if j not in match.forward.values():
                                        not_connected.append(j)
                    connect = False
                    for i in range(len(not_connected)):
                        for j in range(i+1, len(not_connected)):
                            if not_connected[j] in graph.neighbors[not_connected[i]]:
                                connect = True
                    if connect:
                        continue
                if ligand.startswith('Pyrimidazole'):
                    # There is overlap in the structure, so that a phenyl hydrogen
                    # of the building block has both a neighbor from its phenyl ring and
                    # from the pyrimidazole linkage
                    h_pos = system.pos[match.forward[11]]
                    c_pos = system.pos[match.forward[7]]
                    if np.linalg.norm(h_pos - c_pos) < 1.0*angstrom:
                        continue
                # Sometimes, molmod returns a wrong match. Check the match to be sure
                correct = True
                for pattern_edge in pattern.pattern_graph.edges:
                    graph_edge = frozenset([match.forward[key] for key in pattern_edge])
                    if not graph_edge in graph.edges:
                        correct = False
                for i in range(pattern.pattern_graph.num_vertices):
                    if not graph.numbers[match.forward[i]] == pattern.pattern_graph.numbers[i]:
                        correct = False
                if not correct:
                    continue
                # Check that the linkage is not yet present
                ligand_index = [match.forward[key] for key in allowed]
                if any([i in indices for i in ligand_index]):
                    assert all([i in indices for i in ligand_index]), '{} ({}) already occupied'.format(ligand, match.forward.values())
                    continue
                # Extra criterium: the linkage does not create isolated parts, the framework remains connected
                ligand_index = [match.forward[key] for key in allowed]
                subgraph = graph.get_subgraph([i for i in range(graph.num_vertices) if not i in ligand_index], normalize = True)
                parts = get_isolated_parts(subgraph)
                if not n_parts == max(parts) + 1:
                    continue
                
                # Linkage is accepted
                ligand_indices.update(ligand_index)
                new_graph = graph.get_subgraph([i for i in range(graph.num_vertices) if not i in ligand_indices])
                break
            else:
                break
        n_tot = len(ligand_indices)/(len(allowed)*np.prod(reps))
        assert len(ligand_indices) % (len(allowed)*np.prod(reps)) == 0.0
        fn_out.write('{} {}\n'.format(ligand, int(n_tot)))
        indices.update(ligand_indices)
    indices = set([i for i in indices if i < system.natom/np.prod(reps)]) # only indices in unit cell, not supercell
    return indices

def get_bond_linkages(system, fn_out):
    # Search for C-C or C-N bond by partitioning the system in SBUs
    graph = MolecularGraph(system.bonds, system.numbers)
    all_bonds = set([])
    for counter, linkages in enumerate([cc, cn]):
        indices = set([])
        all_bonds_linkage = set([])
        for name in sorted(linkages.keys(), key=lambda e: linkages[e][0].pattern_graph.num_vertices, reverse = True):
            pattern, bonds = linkages[name]
            bonds = np.array(bonds)
            # Search for pattern
            graph_search = GraphSearch(pattern)
            new_graph = graph
            while True:
                for match in graph_search(new_graph):
                    # Sometimes, molmod returns a wrong match. Check the match to be sure
                    correct = True
                    for pattern_edge in pattern.pattern_graph.edges:
                        graph_edge = frozenset([match.forward[key] for key in pattern_edge])
                        if not graph_edge in graph.edges:
                            correct = False
                    for i in range(pattern.pattern_graph.num_vertices):
                        if not graph.numbers[match.forward[i]] == pattern.pattern_graph.numbers[i]:
                            correct = False
                    if not correct:
                        continue
                    # Check that the building block is not yet present
                    # To prevent that e.g. phenyl-rings are identified in a trisphenyl-benzene block
                    building_block = [match.forward[i] for i in range(pattern.pattern_graph.num_vertices) if not i in bonds.flatten()]
                    if any([i in indices for i in building_block]):
                        continue

                    # Match accepted
                    indices.update(building_block)
                    new_graph = graph.get_subgraph([i for i in range(graph.num_vertices) if not i in indices])
                    for bond in bonds:
                        all_bonds_linkage.update([frozenset([match.forward[i] for i in bond])])
                else:
                    break
        all_bonds.update(all_bonds_linkage)
        if counter == 0:
            fn_out.write('C-C {}\n'.format(len(all_bonds_linkage)))
        elif counter == 1:
            fn_out.write('C-N {}\n'.format(len(all_bonds_linkage)))
    return all_bonds



def iter_graphs(system, fn_out):
    natom0 = system.natom
    system, graph, n_parts = clean_system(system)
    ligands = get_ligands(system, fn_out, n_parts)
    linkers = [i for i in range(graph.num_vertices) if i not in ligands]
    linker_graph = graph.get_subgraph(linkers)
    if len(ligands) == 0:
        bond_linkages = get_bond_linkages(system, fn_out)
        graph_edges = list(linker_graph.edges)
        for bond in bond_linkages:
            graph_edges.remove(bond)
        linker_graph = MolecularGraph(graph_edges, system.numbers)
    else:
        fn_out.write('C-C 0\nC-N 0\n')
    connecting = [i for i in linkers if not len(graph.neighbors[i]) == len(linker_graph.neighbors[i])]
    functional = set([])
    for i0, i1 in linker_graph.edges:
        try:
            part0, part1 = graph.get_halfs(i0, i1)
            if any([index in connecting for index in part0]):
                functional_part = list(part1)
            elif any([index in connecting for index in part1]):
                functional_part = list(part0)
            if len(functional_part) == 1 and graph.numbers[functional_part[0]] == 1: continue
            functional.update(functional_part)
        except GraphError as e:
            continue
    functional = list(functional)
    
    yield 'LigandRAC', graph, ligands, [i for i in range(graph.num_vertices)]
    yield 'FullLinkerRAC', linker_graph, linkers, linkers
    yield 'LinkerConnectingRAC', linker_graph, connecting, linkers
    yield 'FunctionalGroupRAC', linker_graph, functional, linkers

def compute_racs(graph, start, scope):
    props = ['I', 'T', 'X', 'S', 'Z', 'a']
    result = {}
    for d in range(4):
        for prop in props:
            for method in ['prod', 'diff']:
                result['_'.join([method, prop, str(d)])] = 0.0

    for i in start:
        for j, d in graph.iter_breadth_first(start = i):
            if d == 4: break
            if j not in scope: continue
            for prop in props:
                if prop == 'I':
                    # Identity
                    prop_i = 1
                    prop_j = 1
                elif prop == 'T':
                    # Connectivity
                    prop_i = len(graph.neighbors[i])
                    prop_j = len(graph.neighbors[j])
                elif prop == 'X':
                    # Electronegativity
                    prop_i = endict[pt[system.numbers[i]].symbol]
                    prop_j = endict[pt[system.numbers[j]].symbol]
                elif prop == 'S':
                    # Covalent radius
                    # Different definition molmod and molsimplify
                    prop_i = pt[system.numbers[i]].covalent_radius
                    prop_j = pt[system.numbers[j]].covalent_radius
                elif prop == 'Z':
                    # Nuclear charge
                    prop_i = system.numbers[i]
                    prop_j = system.numbers[j]
                elif prop == 'a':
                    # Polarizability
                    prop_i = poldict[pt[system.numbers[i]].symbol]
                    prop_j = poldict[pt[system.numbers[j]].symbol]
                result['_'.join(['diff', prop, str(d)])] += float(prop_i - prop_j)/len(start)
                result['_'.join(['prod', prop, str(d)])] += float(prop_i*prop_j)/len(start)
    return result

def get_npart(graph):
    # Return the number of separated parts in the graph
    indices = set([])
    for edge in graph.edges:
        indices.update(edge)
    indices = sorted(indices)
    work = -np.ones(len(indices))
    counter = 0
    while -1 in work:
        counter += 1
        start = -1
        for i in range(len(indices)):
            if work[i] == -1:
                start = indices[i]
                break
        assert not start == -1
        for i, depth in graph.iter_breadth_first(start):
            work[indices.index(i)] = 0
    return counter

def output(fn, name = None, racs = None, nstart = None, npart = None):
    if name is None:
        name = 'RAC'
    if racs is None:
        racs = {}
    line = name
    for method in ['prod', 'diff']:
        for prop in ['I', 'T', 'X', 'S', 'Z', 'a']:
            for d in range(4):
                key = '_'.join([method, prop, str(d)])
                value = racs.get(key)
                if value is None:
                    assert name == 'RAC'
                    line += ' {}'.format(key)
                else:
                    line += ' {}'.format(value)
    if nstart is None:
        line += ' N_start'
    else:
        line += ' {}'.format(int(nstart))
    if npart is None:
        line += ' N_part'
    else:
        line += ' {}'.format(int(npart))
    fn.write(line + '\n')

def iter_struct(database):
    db_path = {
            'redd-coffee': '/path/to/redd-coffee',
            'martin': '/path/to/martin',
            'mercado': '/path/to/mercado',
            'core': '/path/to/core',
            'curated': '/path/to/curated'
            }
    for fn_chk in os.listdir(db_path[database]):
        fn_rac = fn_chk.replace('.chk', '.rac')
        yield fn_chk, fn_rac

if __name__ == '__main__':
    for database in ['redd-coffe', 'martin', 'mercado', 'core', 'curated']:
        for fn_chk, fn_rac in iter_struct(database):
            with open(fn_rac, 'w') as f:
                struct = fn_chk.split('/')[-1]
                f.write(struct + '\n')
                system = System.from_file(fn_chk)
                try:
                    for name, graph, start, scope in iter_graphs(system, fn_rac):
                        if name == 'LigandRAC': output(fn_rac)
                        racs = compute_racs(graph, start, scope)
                        npart = get_npart(graph)
                        output(fn_rac, name, racs, len(start), npart)
                except:
                    print('WARNING: could not determine RACs for {}, probably this is because no linkages are identified'.format(struct))
                    continue


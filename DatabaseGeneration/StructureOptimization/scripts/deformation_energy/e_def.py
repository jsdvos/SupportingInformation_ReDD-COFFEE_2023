import os
import sys
sys.path.insert(1, '../../../StructureAssembly/scripts/')

import time
import numpy as np

from yaff import System, ForceField, ForcePartPair, ForcePartValence, NeighborList, angstrom, kjmol, neigh_dtype
from molmod import MolecularGraph, GraphSearch, EqualPattern, CustomPattern, CriteriaSet, HasAtomNumber

from SBU import SBU
from Topology import Topology, wyckoff_number_to_name

class CustomMolecularPattern(CustomPattern):
    '''
    In a GraphSearch, the atom numbers are forced to be the same
    '''
    def compare(self, vertex0, vertex1, subject_graph):
        return self.pattern_graph.numbers[vertex0] == subject_graph.numbers[vertex1]

def get_ff_kwargs():
    # Return force field arguments
    kwargs = {'rcut': 11.0*angstrom,
              'alpha_scale': 2.86,
              'gcut_scale': 1.0,
              'smooth_ei': True,
              'tailcorrections': True}
    return kwargs

def iter_struct():
    '''
    How to iterate over all structures in the database.
    If new folders are created, implement this here instead of altering the main script
    '''
    src_path = '/path/to/database'
    for link in range(11):
        for dim in ['2D', '3D']:
            folder = os.path.join(src_path, 'Link{}_{}'.folder(link+1, dim))
            for struct in os.listdir(folder):
                fn_sys = os.path.join(folder, struct, struct + '_optimized')
                fn_pars = os.path.join(folder, struct, 'pars_cluster.txt')
                fn_out = fn_sys.replace('.chk', '.eng')
                yield struct, dim, fn_sys, fn_pars, fn_out

def get_sbus():
    sbus = {}
    sbu_path = '../../../StructureAssembly/data/SBUs/'
    for sbu_name in os.listdir(sbu_path):
        for sbu_termination in os.listdir(os.path.join(sbu_path, sbu_name)):
            sbu = SBU.load(sbu_name, sbu_termination)
            sbus[sbu_name + sbu_termination.replace('_', '-')] = sbu
    return sbus
sbus = get_sbus()

def get_tops():
    tops = {}
    top_path = '../../../StructureAssembly/data/Topologies/'
    for dim in ['2D', '3D']:
        for fn in os.listdir(os.path.join(top_path, dim)):
            top_name = fn.split('.')[0]
            tops[top_name] = Topology.load(top_name, dim)
    return tops
tops = get_tops()

def get_nonmatching_ff_terms():
    result = {}
    with open('nonmatching_ff_terms.txt', 'r') as f:
        for line in f:
            if not line.startswith('\t'):
                sbu_key = line.strip()
                result[sbu_key] = {}
            elif not line.startswith('\t\t'):
                prefix = line.strip()
                result[sbu_key][prefix] = []
            else:
                key = line.strip()
                result[sbu_key][prefix].append(key)
    return result
nonmatching_ff_terms = get_nonmatching_ff_terms()

e_sbus = {} # Used as archive to not calculate the same energy every time again
def get_e_sbu(system, fn_pars, ff_kwargs):
    """
    Energy computation of the individual SBUs

    Covalent interactions: scaled with scaling factor according to number of atoms in internal
    Nonbonded interactions: only interactions between atoms in internal
    """
    graph = MolecularGraph(system.bonds, system.numbers)
    i = 0
    e_cov_sbus = 0.0
    e_ei_sbus = 0.0
    e_mm3_sbus = 0.0
    tot_terms = 0.0
    while i < system.natom: # i is the index of the first atom of an SBU
        # STEP 1: Define small environment of the periodic system
        sbu_name = system.get_ffatype(i)[-8:]
        sbu = sbus[sbu_name]

        # Get internal of SBU
        bonds = []
        internal_bonds = []
        numbers = []
        per_ffatypes = []
        per_indices = []
        per_to_sbu = {}
        for j in range(i, i+sbu.sys.natom):
            numbers.append(system.numbers[j])
            per_ffatypes.append(system.get_ffatype(j))
            per_indices.append(j)
            per_to_sbu[j] = len(numbers) - 1
        for bond in sbu.sys.bonds:
            bonds.append(frozenset(bond))
            internal_bonds.append(frozenset([bond[0] + i, bond[1] + i]))
        # Get termination
        for poe in sbu.poes:
            # Clear per_to_term, as every termination has to be independent
            # In some structures, two different terminations correspond to the same SBU
            # If this would be assigned the same atom, no corresponding SBU termination would be found
            per_to_term = {i + poe[1]: per_to_sbu[i + poe[1]]}
            for j, dist, path in graph.iter_breadth_first(start = i + poe[1], do_paths = True, do_duplicates = True):
                # Avoid addition of other terminations that are less than a distance 3 away (e.g.: boroxine)
                if dist > 3: break
                if frozenset(path[:2]) in internal_bonds: continue
                if dist == 0: continue
                if j not in per_to_term.keys():
                    if system.get_ffatype(j).startswith('F_C_02-') or system.get_ffatype(j).startswith('O_HC_12-'):
                        # This atom is not present in the SBU termination
                        # Replace with a hydrogen atom to find the corresponding SBU configuration
                        # By adding a non-existent ffatype, no FF term with this atom will be counted
                        numbers.append(1)
                        per_ffatypes.append('H_None')
                    else:
                        numbers.append(system.numbers[j])
                        per_ffatypes.append(system.get_ffatype(j))
                    per_indices.append(j)
                    per_to_term[j] = len(numbers) - 1
                bond = frozenset([per_to_term[k] for k in path[-2:]])
                if not bond in bonds:
                    bonds.append(bond)
        # Create graph and subsystem (subsystem only needed for debugging)
        graph0 = MolecularGraph(bonds, numbers)
        per_sys = system.subsystem(per_indices) # Be aware that some indices can be present twice in per_indices

        # STEP 2: Define small environment of the SBU
        sbu_graph = MolecularGraph(sbu.full_sys.bonds, sbu.full_sys.numbers)
        environment_sbu = [j for j in range(sbu.full_sys.natom) if (not sbu.full_sys.get_ffatype(j).endswith('_term') or int(sbu.full_sys.get_ffatype(j).split('_')[0][-1]) <= 3)] # All atoms in internal and termination at max distance 3 from internal
        numbers = []
        bonds = []
        sbu_to_graph = {}
        for j in environment_sbu:
            numbers.append(sbu.full_sys.numbers[j])
            sbu_to_graph[j] = len(numbers) - 1
        for bond in sbu.full_sys.bonds:
            if bond[0] in environment_sbu and bond[1] in environment_sbu:
                ffatype0 = sbu.full_sys.get_ffatype(bond[0])
                ffatype1 = sbu.full_sys.get_ffatype(bond[1])
                if not ffatype0.endswith('_term') or not ffatype1.endswith('_term'):
                    bonds.append(frozenset([sbu_to_graph[bond[0]], sbu_to_graph[bond[1]]]))
                elif abs(int(ffatype0.split('_')[0][-1]) - int(ffatype1.split('_')[0][-1])) == 1:
                    # Avoid adding bonds that are not connected to internal SBU
                    # E.g.: boronate ester termination, no C3-C3 bond
                    bonds.append(frozenset([sbu_to_graph[bond[0]], sbu_to_graph[bond[1]]]))
        graph1 = MolecularGraph(bonds, numbers)
        sbu_sys = sbu.full_sys.subsystem(environment_sbu)

        # STEP 3: Map SBU indices onto indices from periodic
        # subsystem and identify termination ffatypes
        
        # Define SBU pattern that we search in small environment of periodic system
        pattern = CustomMolecularPattern(graph1) # CustomMolecularPattern also checks atom number
        graph_search = GraphSearch(pattern)
        matches = list(graph_search(graph0, one_match = True)) # One match is sufficient
        
        per_to_sbu = matches[0].forward
        ffatypes = []
        sbu_indices = []
        for j in range(sbu_sys.natom):
            if not sbu_sys.get_ffatype(j).endswith('_term'):
                sbu_indices.append(j)
            ffatypes.append(per_ffatypes[per_to_sbu[j]])
        # Overwrite the ffatypes of the small SBU system
        sbu_sys.ffatype_ids = None
        sbu_sys.ffatypes = ffatypes
        sbu_sys._init_derived_ffatypes()

        # STEP 4: Calculate the energy (ForcePartPairs with adapted neighborlist, 
        # ForcePartValence with scaling for overlap terms)
        sbu_ff_key = '.'.join(ffatypes)
        if not sbu_ff_key in e_sbus.keys():
            sbu_ff = ForceField.generate(sbu_sys, fn_pars, **ff_kwargs)
            sbu_ff.compute()
            # Define new NeighborList with only interactions between atoms in the internal SBU
            nlist = NeighborList(sbu_sys)
            for j in range(sbu_ff.nlist.nneigh):
                neigh = sbu_ff.nlist.neighs[j]
                if neigh[0] in sbu_indices and neigh[1] in sbu_indices:
                    nlist.neighs[nlist.nneigh] = neigh
                    nlist.nneigh += 1
                    if nlist.nneigh == len(nlist.neighs):
                        last_start = len(nlist.neighs)
                        new_neighs = np.empty(len(nlist.neighs)*3//2, dtype = neigh_dtype)
                        new_neighs[:last_start] = nlist.neighs
                        nlist.neighs = new_neighs
            
            e_cov_sbu = 0.0
            e_ei_sbu = 0.0
            e_mm3_sbu = 0.0
            for part in sbu_ff.parts:
                if isinstance(part, ForcePartPair):
                    # Add energy with the new NeighborList
                    if part.name == 'pair_ei':
                        e_ei_sbu += part.pair_pot.compute(nlist.neighs, part.scalings.stab, None, None, nlist.nneigh)
                    elif part.name == 'pair_mm3':
                        e_mm3_sbu += part.pair_pot.compute(nlist.neighs, part.scalings.stab, None, None, nlist.nneigh)
                elif isinstance(part, ForcePartValence):
                    sbu_scale = 0.0
                    for j in range(part.vlist.nv):
                        vterm = part.vlist.vtab[j]
                        atoms = part.vlist.lookup_atoms(j)
                        if part.iclist.ictab[vterm['ic0']]['kind'] == 0 and vterm['ic1'] == -1:
                            prefix = 'BONDHARM'
                            atom_indices = (atoms[0][0][0], atoms[0][0][1])
                        elif part.iclist.ictab[vterm['ic0']]['kind'] == 2 and vterm['ic1'] == -1:
                            prefix = 'BENDAHARM'
                            assert atoms[0][0][0] == atoms[0][1][0]
                            atom_indices =  (atoms[0][0][1], atoms[0][0][0], atoms[0][1][1])
                        elif part.iclist.ictab[vterm['ic0']]['kind'] in [3, 4] and vterm['ic1'] == -1:
                            # See TorsionGenerator.get_vterm: in general, kind0 == 4 (DihedAngle),
                            # but mostly kind0 == 3 (DihedCos) and a Chebychev is used
                            prefix = 'TORSION'
                            assert atoms[0][0][0] == atoms[0][1][0]
                            assert atoms[0][1][1] == atoms[0][2][0]
                            atom_indices = (atoms[0][0][1], atoms[0][1][0], atoms[0][1][1], atoms[0][2][1])
                        elif part.iclist.ictab[vterm['ic0']]['kind'] == 10 and vterm['ic1'] == -1:
                            prefix = 'OOPDIST'
                            assert atoms[0][0][1] == atoms[0][1][0]
                            assert atoms[0][1][1] == atoms[0][2][0]
                            atom_indices = (atoms[0][0][0], atoms[0][0][1], atoms[0][1][1], atoms[0][2][1])
                        elif part.iclist.ictab[vterm['ic0']]['kind'] == 0 and part.iclist.ictab[vterm['ic1']]['kind'] == 0:
                            prefix = 'CROSS'
                            assert atoms[0][0][1] == atoms[1][0][0]
                            atom_indices = (atoms[0][0][0], atoms[0][0][1], atoms[1][0][1])
                        elif part.iclist.ictab[vterm['ic0']]['kind'] == 0 and part.iclist.ictab[vterm['ic1']]['kind'] == 2:
                            prefix = 'CROSS'
                            assert atoms[1][0][0] == atoms[1][1][0]
                            assert atoms[0][0] == atoms[1][0][::-1] or atoms[0][0] == atoms[1][1]
                            atom_indices =  (atoms[1][0][1], atoms[1][0][0], atoms[1][1][1])
                        else:
                            kind0 = part.iclist.ictab[vterm['ic0']]['kind']
                            try:
                                kind1 = vterm['ic1']['kind']
                            except:
                                kind1 = -1
                            print('Atom indices not yet defined for kinds {} and {} in FF from {}'.format(kind0, kind1, fn_pars))
                        # Find other_sbu_name
                        other_sbu_name = 'None'
                        for k in atom_indices:
                            if k not in sbu_indices:
                                ffatype = sbu_sys.get_ffatype(k)
                                assert other_sbu_name in ['None', ffatype[-8:]]
                                other_sbu_name = ffatype[-8:]
                        
                        scale = float(sum(k in sbu_indices for k in atom_indices))/len(atom_indices)
                        key = '.'.join([sbu_sys.get_ffatype(k) for k in atom_indices])
                        sbu_names = [sbu_name, other_sbu_name]
                        for names, corrected_scale in [[sbu_names, 1.0], [sbu_names[::-1], 0.0]]:
                            sbu_key = '+'.join(names)
                            if sbu_key in nonmatching_ff_terms.keys():
                                lines = nonmatching_ff_terms[sbu_key].get(prefix)
                                if not lines is None:
                                    if key in lines:
                                        scale = corrected_scale
                        sbu_scale += scale
                        e_cov_sbu += scale*vterm['energy']
                else:
                    raise RuntimeError('Didnt expect a {} ForcePart in an SBU force field'.format(part.name))
            e_sbus[sbu_ff_key] = (e_cov_sbu, e_ei_sbu, e_mm3_sbu, sbu_scale)
        e_cov_sbu, e_ei_sbu, e_mm3_sbu, sbu_scale = e_sbus[sbu_ff_key]
        e_cov_sbus += e_cov_sbu
        e_ei_sbus += e_ei_sbu
        e_mm3_sbus += e_mm3_sbu
        tot_terms += sbu_scale
        i += sbu.sys.natom
    return e_cov_sbus, e_ei_sbus, e_mm3_sbus, tot_terms

def get_e_local(system, fn_pars, ff_kwargs):
    """
    Energy calculation of the interactions between (inter) and within (intra) SBUs

    Covalent interactions are included fully
    Nonbonded interactions: only interactions between atoms in the same SBU
    """
    # Step 0: Define local force field without Gaussian screening charge correction (alpha = 0.0)
    if 'alpha_scale' in ff_kwargs.keys():
        ff_kwargs['alpha_scale'] = 0.0
    ff_kwargs['reci_ei'] = 'ignore' # Reciprocal interactions with alpha = 0.0 raise ZeroDivisionError
    ff = ForceField.generate(system, fn_pars, **ff_kwargs)
    ff.compute()
    # STEP 1: Partition atoms by SBU
    partitions = [] # List of partitions, each atom is present in exactly one partition
    partition_indices = [] # List of partition indices in which each atom is present
    i = 0
    count = 0
    while i < ff.system.natom:
        sbu_name = ff.system.get_ffatype(i)[-8:]
        partitions.append([])
        for j in range(sbus[sbu_name].sys.natom):
            partition_indices.append(count)
            partitions[-1].append(i + j)
        i += sbus[sbu_name].sys.natom
        count += 1
    # STEP 2: Define adapted neighbor list containing
    # only local interactions (intra) or excluding local interactions (inter)
    intra_nlist = NeighborList(ff.system)
    inter_nlist = NeighborList(ff.system)
    for i in range(ff.nlist.nneigh):
        neigh = ff.nlist.neighs[i]
        dist = np.linalg.norm(ff.system.pos[neigh[1]] - ff.system.pos[neigh[0]])
        if neigh[1] in partitions[partition_indices[neigh[0]]] and abs(neigh['d'] - dist) < 0.01*angstrom:
            # Add to intra nlist
            intra_nlist.neighs[intra_nlist.nneigh] = neigh
            intra_nlist.nneigh += 1
            if intra_nlist.nneigh == len(intra_nlist.neighs):
                # Make nlist longer when needed
                last_start = len(intra_nlist.neighs)
                new_neighs = np.empty(len(intra_nlist.neighs)*3//2, dtype = neigh_dtype)
                new_neighs[:last_start] = intra_nlist.neighs
                intra_nlist.neighs = new_neighs
        else:
            # Add to inter nlist
            inter_nlist.neighs[inter_nlist.nneigh] = neigh
            inter_nlist.nneigh += 1
            if inter_nlist.nneigh == len(inter_nlist.neighs):
                # Make nlist longer when needed
                last_start = len(inter_nlist.neighs)
                new_neighs = np.empty(len(inter_nlist.neighs)*3//2, dtype = neigh_dtype)
                new_neighs[:last_start] = inter_nlist.neighs
                inter_nlist.neighs = new_neighs
    # STEP 3: Calculate the energy with the adapted neighbor lists
    cov = 0.0
    ei_intra = 0.0
    ei_inter = 0.0
    vdw_intra = 0.0
    vdw_inter = 0.0
    for part in ff.parts:
        if isinstance(part, ForcePartPair):
            if part.name == 'pair_ei':

                ei_intra = part.pair_pot.compute(intra_nlist.neighs, part.scalings.stab, None, None, intra_nlist.nneigh)
                ei_inter = part.pair_pot.compute(inter_nlist.neighs, part.scalings.stab, None, None, inter_nlist.nneigh)
            elif part.name == 'pair_mm3':
                vdw_intra = part.pair_pot.compute(intra_nlist.neighs, part.scalings.stab, None, None, intra_nlist.nneigh)
                vdw_inter = part.pair_pot.compute(inter_nlist.neighs, part.scalings.stab, None, None, inter_nlist.nneigh)
        elif isinstance(part, ForcePartValence):
            cov = part.energy
            tot_terms = part.vlist.nv
    return cov, ei_intra, ei_inter, vdw_intra, vdw_inter, tot_terms

def analyze_struct(struct):
    # Analyze how many SBUs and linkages are present within the structure
    words = struct.split('_')
    top_name = words[0]
    sbu_names = words[1:]
    top = tops[top_name]
    if sbu_names[0] == '2':
        sbu_names_correct = []
        for i in range(len(top.wyckoff_vertices)):
            wyckoff_name = wyckoff_number_to_name(i)
            wyckoff_vertex = top.wyckoff_vertices[wyckoff_name]
            cn = wyckoff_vertex.cn
            sbu_names_correct.append(sbu_names[sbu_names.index(str(cn))+1])
        for i in range(len(top.wyckoff_edges)):
            cn = 2
            sbu_names_correct.append(sbu_names[sbu_names.index(str(cn))+1])
        sbu_names = sbu_names_correct
    nlink = 0
    nsbu = 0
    for i in range(len(top.wyckoff_vertices)):
        wyckoff_name = wyckoff_number_to_name(i)
        wyckoff_vertex = top.wyckoff_vertices[wyckoff_name]
        nsbu += len(top.wyckoff_vertices[wyckoff_name])
    for i in range(len(top.wyckoff_edges)):
        wyckoff_name = wyckoff_number_to_name(i, True)
        wyckoff_edge = top.wyckoff_edges[wyckoff_name]
        sbu_name = sbu_names[len(top.wyckoff_vertices) + i]
        n_nodes = len(top.wyckoff_edges[wyckoff_name])
        if sbu_name == 'None':
            nlink += n_nodes
        else:
            nlink += 2*n_nodes
            nsbu += n_nodes
    return nlink, nsbu

if __name__ == '__main__':
    # This is more efficient to run for each linkage type, than for each structure
    # E_sbu is mostly the same and should not be computed for each structure
    # Instead it is stored and retrieved once needed
    for struct, dim, fn_sys, fn_pars, fn_out in iter_struct():
        system = System.from_file(fn_sys)
        ff_kwargs = get_ff_kwargs()
        ff = ForceField.generate(system, fn_pars, **ff_kwargs)
        natom = system.natom
        nlink, nsbu = analyze_struct(struct) # Should be multiplied with 2 for 2-layer systems
        e_tot = ff.compute()
        for part in ff.parts:
            if part.name == 'pair_ei':
                e_ei = part.energy
            elif part.name == 'ewald_reci':
                e_ei_reci = part.energy
            elif part.name == 'ewald_cor':
                e_ei_cor = part.energy
            elif part.name == 'ewald_neut':
                e_ei_neut = part.energy
            elif part.name == 'pair_mm3':
                e_mm3 = part.energy
            elif part.name == 'tailcorr_pair_mm3':
                e_mm3_tail = part.energy
        e_cov, e_ei_intra, e_ei_inter, e_mm3_intra, e_mm3_inter, tot_terms_per = get_e_local(system, fn_pars, ff_kwargs)
        e_ei_scr = e_ei - e_ei_intra - e_ei_inter
        assert abs(e_mm3_intra + e_mm3_inter - e_mm3) < 1e-9
        assert abs(e_cov + e_ei_intra + e_ei_inter + e_ei_scr + e_ei_reci + e_ei_cor + e_ei_neut + e_mm3_intra + e_mm3_inter + e_mm3_tail - e_tot) < 1e-9
        e_cov_sbu, e_ei_sbu, e_mm3_sbu, tot_terms_sbu = get_e_sbu(system, fn_pars, ff_kwargs)
        assert abs(round(tot_terms_sbu, 6) - round(tot_terms_per, 6)) < 1e-9, '{} != {}'.format(tot_terms_sbu, tot_terms_per)
        with open(fn_out, 'a') as f:
            f.write('{} {} {} {} {} {} {} {} {} {} {}\n'.format(struct, e_cov, e_ei_intra, e_ei_inter, e_ei_scr, e_ei_reci, e_ei_cor, e_ei_neut, e_mm3_intra, e_mm3_inter, e_mm3_tail))
            f.write('{} {} {}\n'.format(e_cov_sbu, e_ei_sbu, e_mm3_sbu))
            f.write('{} {} {}\n'.format(natom, nlink, nsbu))
        if dim == '2D':
            # Include 2 layers
            nlink *= 2
            nsbu *= 2
        e_deformation = (e_per_cov + e_per_ei_intra + e_per_mm3_intra - e_sbu_cov - e_sbu_ei - e_sbu_mm3)/nlink
        with open('../../data/database_edef.txt', 'a') as f:
            f.write('{} {}\n'.format(struct, e_deformation))


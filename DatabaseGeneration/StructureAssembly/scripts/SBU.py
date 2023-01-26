#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from molmod.units import angstrom, kjmol
from molmod.periodic import periodic
from molmod.transformations import Translation, Rotation, Complete, superpose, fit_rmsd
from yaff import System, log
from yaff.pes.ff import ForceField, ForcePartPair, ForcePartValence
from yaff.pes.nlist import NeighborList
from yaff.pes.ext import neigh_dtype
from yaff import XYZWriter
from yaff.sampling.dof import CartesianDOF
from yaff.sampling.opt import CGOptimizer
from copy import deepcopy
from ParametersCombination import ParametersCombination
log.set_level(0)

class SBU():
    sbu_path = '../data/SBUs/'
    def __init__(self, name, environment, termination, sys, center, poes, full_sys = None):
        '''
        A Secondary Building Unit (SBU) is the building block of the framework. It is assumed to remain rigid in the sense that
        its internal geometry doesn't change when put in a different environment. Therefore, it can be simplified by only considering
        its points of extension (PoE), which are the positions that point towards neighboring SBUs. By fixing these points of extension,
        the whole internal geometry of the SBU is fixed. Besides, by assigning an atom to the PoE, a bond between two atoms of different
        SBUs can be obtained by finding the PoEs of both SBUs that connect to each other.

        Attributes:
            name [string]: Name of the SBU
            termination [string]: indicates how the SBU FF is terminated, how the environment is described
            sys [yaff System]: System of the atoms present in the SBU
            center [numpy array]: The center of the SBU (not necessarily the com), neighboring SBUs will point towards this center
            poes: Points of Extension of the SBU. Every point of extension consists of three elements:
                    poe[0] [numpy array]: position of the point of extension
                    poe[1] [integer]: index of the atom that is assigned to the point of extension
                    poe[2] [SBU]: SBU that is connected with the point of extension [default = None]
            parameters [ParametersCombination]: Force field parameters to describe the interactions between the atoms in the SBU


        TO DO: give more flexibility to initiate an SBU
        TO DO: also give internal_end_group argument, which assigns the termination that neighboring SBUs need if they want to bond with this SBU
        '''
        self.name = name
        self.environment = environment
        self.termination = termination
        self.sys = sys
        self.full_sys = full_sys
        self.center = center
        self.poes = poes
        self.parameters = ParametersCombination()
        self.uff_parameters = ParametersCombination()

    def __str__(self):
        text = 'SBU {} ({} connected, {} atoms)\n\n'.format(self.name, len(self.poes), self.sys.natom)
        text += 'CENTER - {}\n'.format(self.center)
        for poe in self.poes:
            text += 'POE - {} (-> ATOM {})\n'.format(poe[0], poe[1])
        text += '\n'
        for i in range(self.sys.natom):
            text += 'ATOM {} - {} ({}): {}\n'.format(i, periodic[self.sys.numbers[i]].symbol, self.sys.ffatypes[self.sys.ffatype_ids[i]], self.sys.pos[i])
        return text

    @classmethod
    def load(cls, name, termination):
        '''
        An SBU is loaded from two input files, which are present in the folder sbu_path/name/termination/

        name_termination.chk
        A yaff CHK-file containing the geometry of the SBU and its atom types. The atom types define the difference between the internal geometry
        and the termination. The atoms that are present in the internal of the SBU are assigned an atom type beginning with its element symbol,
        followed by a '_' and ending on '_SBU', where SBU is the name of the SBU (Example: C_B_PDBA, for a carbon atom in the PDBA SBU). The atoms
        that are present in the termination are assigned an atom type beginning with its element symbol, followed by a number, typically the number
        of bonds to an atom in the internal geometry, and ending on '_term' (Example: C3_term).

        name_termination.sbu
        An SBU-file (extension .sbu) with all the additional information on the SBU. The first line contains the position of the center of the SBU [in angstrom].
        The second line contains the number of PoEs. The following lines each begin with an atom type followed by the position of the PoE. The atom type
        is used to find the atom that is assigned to the PoE.
        '''
        sys = System.from_file(cls.sbu_path + '{}/{}/{}{}.chk'.format(name, termination, name, termination))
        internal_sys = cls._get_internal_sys(sys)
        environment, center, poes = cls._read_sbu_file(cls.sbu_path + '{}/{}/{}{}.sbu'.format(name, termination, name, termination), internal_sys)
        return cls(name, environment, termination, internal_sys, center, poes, full_sys = sys)

    @staticmethod
    def _get_internal_sys(sys):
        '''
        Returns a System with the internal geometry, which is defined by all the atoms from which the atom type does not end with '_term'
        '''
        internal_indices = []
        for i, ffatype_id in enumerate(sys.ffatype_ids):
            if not sys.ffatypes[ffatype_id].split('_')[-1] == 'term':
                # atom is in internal SBU
                internal_indices.append(i)
        return sys.subsystem(internal_indices)

    @staticmethod
    def _read_sbu_file(fn, sys):
        '''
        Reads the .sbu-file fn. This file has the following format:

        Line 1: environment
        Line 2: Position of the center of the SBU [in angstrom]
        Line 3: Number of PoEs = npoes
        Lines 4 to 3+npoes: Atom type + position of the PoE [in angstrom]

        Example of such an SBU-file (PDBA_boroxine.sbu):

        _benzene
        -0.0000189  0.000000  0.000000
        2
        C_B_PDBA   -0.000023999  4.31278698  0.00000
        C_B_PDBA   -0.000025499 -4.31278698  0.00000

        The atom that is assigned to each PoE is defined as the atom with the give atom type that is closest to the given position
        '''
        with open(fn) as f:
            lines = f.readlines()
            environment = lines[0].strip()
            center = np.array([float(coord)*angstrom for coord in lines[1].strip().split()])
            npoes = int(lines[2].strip())
            poes = []
            for i in range(npoes):
                poe_nei_ffatype, poe_pos_x, poe_pos_y, poe_pos_z = lines[i + 3].strip().split()
                poe_pos = np.array([float(poe_pos_x), float(poe_pos_y), float(poe_pos_z)])*angstrom
                ids = np.where(sys.ffatypes == poe_nei_ffatype)[0]
                indices = [j for j, ffatype_id in enumerate(sys.ffatype_ids) if ffatype_id == ids[0]]
                min_distance = None
                min_index = None
                for index in indices:
                    distance = np.linalg.norm(sys.pos[index] - poe_pos)
                    if min_distance == None or distance < min_distance:
                        min_distance = distance
                        min_index = index
                poes.append([poe_pos, min_index, None])
        return environment, center, poes

    def load_parameters(self, fns = None, uff_fns = None):
        '''
        Load the standard parameters from the ff_pars folder in the respective SBU folder
        '''
        if fns == None:
            fns = ['pars_cov_quickff.txt', 'pars_ei_mbis.txt', 'pars_vdw_mm3.txt']
        if uff_fns == None:
            uff_fns = ['pars_cov_uff.txt', 'pars_ei_mbis.txt', 'pars_vdw_uff.txt']
        sbu_pars = ParametersCombination.load(self, fns)
        sbu_uff_pars = ParametersCombination.load(self, uff_fns)
        self.parameters.add_parameters(sbu_pars, internal = True, mixed = True, termination = False)
        self.uff_parameters.add_parameters(sbu_uff_pars, internal = True, mixed = True, termination = False)

    def add_parameters_from_file(self, fn_pars):
        '''
        Read parameters from files and add them to the parameters attribute
        Argument:
            fn_pars can either be a filename or a list of filenames
        '''
        if isinstance(fn_pars, str):
            fn_pars = [fn_pars]
        sbu_parameters = ParametersCombination.from_file(fn_pars)
        self.parameters.add_parameters(sbu_parameters, internal = True, mixed = True, termination = False)

    def match(self, other_sbu):
        '''
        Check if the other SBU can be connected with this SBU
        '''
        functional, linkage = self.environment.split('_')
        other_functional, other_linkage = other_sbu.environment.split('_')
        if linkage == 'Azine':
            return linkage == other_linkage and functional == other_functional
        else:
            return linkage == other_linkage and not functional == other_functional

    def get_radius(self):
        '''
        Returns the radius of an SBU, which is defined as the mean of the distances between the PoE and the center of the SBU
        '''
        radii = []
        for poe in self.poes:
            radii.append(np.linalg.norm(poe[0] - self.center))
        return np.mean(radii)

    def translate(self, t):
        '''
        Translates the SBU over a translation vector t
        '''
        translation = Translation(t)
        self.apply(translation)

    def rotate_about_center(self, angle, axis):
        '''
        Rotates the SBU about an angle around the axis, so that the center of the SBU is not moved
        '''
        rotation = Complete.about_axis(self.center, angle, axis)
        self.apply(rotation)

    def kabsch(self, r_new, r_old, apply = True, only_rotation = False):
        '''
        Finds the transformation so that the old vectors in r_old are maped as close as possible onto the new vectors r_new, while the SBU stays at the same place
        The RMSD is returned, which is defined as the root-mean-square deviation between the new vectors and the transformed old vectors
        If apply is True, the transformation is also effectively applied on the SBU.
        '''
        transformation, r_old_trans, rmsd = fit_rmsd(r_new, r_old)
        if only_rotation:
            transformation = Rotation(transformation.r)
        if apply:
            transformation = Translation(self.center) * transformation * Translation(-self.center)
            self.apply(transformation)
        return rmsd

    def orient(self, nei_pos, only_rotation = False):
        '''
        Transforms the SBU in such a way that the PoEs are oriented towards the positions given in nei_pos
        Every i-th PoE is oriented towards the i-th position in nei_pos
        '''
        poe_pos = []
        for poe in self.poes:
            poe_pos.append((poe[0] - self.center)/np.linalg.norm(poe[0] - self.center))
        
        rmsd = self.kabsch(np.array(nei_pos), np.array(poe_pos), only_rotation = only_rotation)
        for n_pos, poe in zip(nei_pos, self.poes):
            p_pos = poe[0] - self.center
            v1_u = n_pos / np.linalg.norm(n_pos)
            v2_u = p_pos / np.linalg.norm(p_pos)
            angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return rmsd

    def apply(self, transformation):
        '''
        Applies a transformation on the whole SBU
        '''
        self.center = transformation * self.center
        for poe in self.poes:
            poe[0] = transformation * poe[0]
        for i in range(self.sys.natom):
            self.sys.pos[i] = transformation * self.sys.pos[i]

    def copy(self):
        '''
        Returns an independent copy of the SBU
        '''
        new_poes = []
        for poe in self.poes:
            new_poes.append([np.copy(poe[0]), poe[1], poe[2]])
        result = SBU(self.name, self.environment, self.termination, self.sys.subsystem(range(self.sys.natom)), np.copy(self.center), new_poes)
        result.parameters = self.parameters.copy()
        result.uff_parameters = self.uff_parameters.copy()
        return result

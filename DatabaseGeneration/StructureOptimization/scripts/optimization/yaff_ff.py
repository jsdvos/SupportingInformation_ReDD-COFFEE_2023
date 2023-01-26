# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
'''Force field models

   This module contains the force field computation interface that is used by
   the :mod:`yaff.sampling` package.

   The ``ForceField`` class is the main item in this module. It acts as
   container for instances of subclasses of ``ForcePart``. Each ``ForcePart``
   subclass implements a typical contribution to the force field energy, e.g.
   ``ForcePartValence`` computes covalent interactions, ``ForcePartPair``
   computes pairwise (non-bonding) interactions, and so on. The ``ForceField``
   object also contains one neighborlist object, which is used by all
   ``ForcePartPair`` objects. Actual computations are done through the
   ``compute`` method of the ``ForceField`` object, which calls the ``compute``
   method of all the ``ForceParts`` and adds up the results.
'''


from __future__ import division

import numpy as np

from yaff.pes.ext import compute_ewald_reci, compute_ewald_reci_dd, compute_ewald_corr, Switch3, \
    compute_ewald_corr_dd, PairPotEI, PairPotEIDip, PairPotLJ, PairPotLJCross, PairPotMM3, PairPotMM3CAP, PairPotGrimme, compute_grid3d, \
    neigh_dtype, nlist_status_init, nlist_status_finish, nlist_build, nlist_recompute

from yaff.pes.dlist import DeltaList
from yaff.pes.iclist import InternalCoordinateList
from yaff.pes.vlist import ValenceList
from yaff.pes.scaling import Scalings
from yaff.pes.generator import Generator,NonbondedGenerator


class NeighborList(object):
    '''Algorithms to keep track of all pair distances below a given rcut
    '''
    def __init__(self, system, skin=0, nlow=0, nhigh=-1, log=None, timer=None):
        """
           **Arguments:**
           system
                A System instance.
           **Optional arguments:**
           skin
                A margin added to the rcut parameter. Only when atoms are
                displaced by half this distance, the neighbor list is rebuilt
                from scratch. In the other case, the distances of the known
                pairs are just recomputed. If set to zero, the default, the
                neighbor list is rebuilt at each update.
                A reasonable skin setting can drastically improve the
                performance of the neighbor list updates. For example, when
                ``rcut`` is ``10*angstrom``, a ``skin`` of ``2*angstrom`` is
                reasonable. If the skin is set too large, the updates will
                become very inefficient. Some tuning of ``rcut`` and ``skin``
                may be beneficial.
            nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.
            nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
                If nlow=nhigh, the system is divided into two parts and only
                pairs involving one atom of each part will be included. This is
                useful to calculate interaction energies in Monte Carlo
                simulations
        """
        if skin < 0:
            raise ValueError('The skin parameter must be positive.')
        if nhigh == -1:
            nhigh = system.natom
        self.system = system
        self.skin = skin
        self.rcut = 0.0
        # the neighborlist:
        self.neighs = np.empty(10, dtype=neigh_dtype)
        self.nneigh = 0
        self.rmax = None
        if nlow < 0:
            raise ValueError('nlow must be a positive number, received %d.'%nlow)
        self.nlow = nlow
        if nhigh < self.nlow:
            raise ValueError('nhigh must not be smaller than nlow, received %d.'%nhigh)
        self.nhigh = nhigh
        # for skin algorithm:
        self._pos_old = None
        self.rebuild_next = False
        self.log = log
        self.timer = timer

    def request_rcut(self, rcut):
        """Make sure the internal rcut parameter is at least is high as rcut."""
        self.rcut = max(self.rcut, rcut)
        self.update_rmax()

    def update_rmax(self):
        """Recompute the ``rmax`` attribute.
           ``rmax`` determines the number of periodic images that are
           considered. when building the neighbor list. Along the a direction,
           images are taken from ``-rmax[0]`` to ``rmax[0]`` (inclusive). The
           range of images along the b and c direction are controlled by
           ``rmax[1]`` and ``rmax[2]``, respectively.
           Updating ``rmax`` may be necessary for two reasons: (i) the cutoff
           has changed, and (ii) the cell vectors have changed.
        """
        # determine the number of periodic images
        self.rmax = np.ceil((self.rcut+self.skin)/self.system.cell.rspacings-0.5).astype(int)
        if self.log.do_high:
            if len(self.rmax) == 1:
                self.log('rmax a       = %i' % tuple(self.rmax))
            elif len(self.rmax) == 2:
                self.log('rmax a,b     = %i,%i' % tuple(self.rmax))
            elif len(self.rmax) == 3:
                self.log('rmax a,b,c   = %i,%i,%i' % tuple(self.rmax))
        # Request a rebuild of the neighborlist because there is no simple way
        # to figure out whether an update is sufficient.
        self.rebuild_next = True

    def update(self):
        '''Rebuild or recompute the neighbor lists
           Based on the changes of the atomic positions or due to calls to
           ``update_rcut`` and ``update_rmax``, the neighbor lists will be
           rebuilt from scratch.
           The heavy computational work is done in low-level C routines. The
           neighbor lists array is reallocated if needed. The memory allocation
           is done in Python for convenience.
        '''
        with self.log.section('NLIST'), self.timer.section('Nlists'):
            assert self.rcut > 0

            if self._need_rebuild():
                # *rebuild* the entire neighborlist
                if self.system.cell.volume != 0:
                    if self.system.natom/self.system.cell.volume > 10:
                        raise ValueError('Atom density too high')
                # 1) make an initial status object for the neighbor list algorithm
                status = nlist_status_init(self.rmax)
                # The atom index of the first atom in pair is always at least
                # nlow. The following status initialization avoids searching
                # for excluded atom pairs in the neighbourlist build
                status[3] = self.nlow
                # 2) a loop of consecutive update/allocate calls
                last_start = 0
                while True:
                    done = nlist_build(
                        self.system.pos, self.rcut + self.skin, self.rmax,
                        self.system.cell, status, self.neighs[last_start:], self.nlow, self.nhigh
                    )
                    if done:
                        break
                    last_start = len(self.neighs)
                    new_neighs = np.empty((len(self.neighs)*3)//2, dtype=neigh_dtype)
                    new_neighs[:last_start] = self.neighs
                    self.neighs = new_neighs
                    del new_neighs
                # 3) get the number of neighbors in the list.
                self.nneigh = nlist_status_finish(status)
                if self.log.do_debug:
                    self.log('Rebuilt, size = %i' % self.nneigh)
                # 4) store the current state to check in future calls if we
                #    need to do a rebuild or a recompute.
                self._checkpoint()
                self.rebuild_next = False
            else:
                # just *recompute* the deltas and the distance in the
                # neighborlist
                nlist_recompute(self.system.pos, self._pos_old, self.system.cell, self.neighs[:self.nneigh])
                if self.log.do_debug:
                    self.log('Recomputed')

    def _checkpoint(self):
        '''Internal method called after a neighborlist rebuild.'''
        if self.skin > 0:
            # Only use the skin algorithm if this parameter is larger than zero.
            if self._pos_old is None:
                self._pos_old = self.system.pos.copy()
            else:
                self._pos_old[:] = self.system.pos

    def _need_rebuild(self):
        '''Internal method that determines if a rebuild is needed.'''
        if self.skin <= 0 or self._pos_old is None or self.rebuild_next:
            return True
        else:
            # Compute an upper bound for the maximum relative displacement.
            disp = np.sqrt(((self.system.pos - self._pos_old)**2).sum(axis=1).max())
            disp *= 2*(self.rmax.max()+1)
            if self.log.do_debug:
                self.log('Maximum relative displacement %s      Skin %s' % (self.log.length(disp), self.log.length(self.skin)))
            # Compare with skin parameter
            return disp >= self.skin


    def to_dictionary(self):
        """Transform current neighbor list into a dictionary.
           This is slow. Use this method for debugging only!
        """
        dictionary = {}
        for i in range(self.nneigh):
            key = (
                self.neighs[i]['a'], self.neighs[i]['b'], self.neighs[i]['r0'],
                self.neighs[i]['r1'], self.neighs[i]['r2']
            )
            value = np.array([
                self.neighs[i]['d'], self.neighs[i]['dx'],
                self.neighs[i]['dy'], self.neighs[i]['dz']
            ])
            dictionary[key] = value
        return dictionary


    def check(self):
        """Perform a slow internal consistency test.
           Use this for debugging only. It is assumed that self.rmax is set correctly.
        """
        # 0) Some initial tests
        assert (
            (self.neighs['a'][:self.nneigh] > self.neighs['b'][:self.nneigh]) |
            (self.neighs['r0'][:self.nneigh] != 0) |
            (self.neighs['r1'][:self.nneigh] != 0) |
            (self.neighs['r2'][:self.nneigh] != 0)
        ).all()
        # A) transform the current nlist into a set
        actual = self.to_dictionary()
        # B) Define loops of cell vectors
        if self.system.cell.nvec == 3:
            def rloops():
                for r2 in range(0, self.rmax[2]+1):
                    if r2 == 0:
                        r1_start = 0
                    else:
                        r1_start = -self.rmax[1]
                    for r1 in range(r1_start, self.rmax[1]+1):
                        if r2 == 0 and r1 == 0:
                            r0_start = 0
                        else:
                            r0_start = -self.rmax[0]
                        for r0 in range(r0_start, self.rmax[0]+1):
                            yield r0, r1, r2
        elif self.system.cell.nvec == 2:
            def rloops():
                for r1 in range(0, self.rmax[1]+1):
                    if r1 == 0:
                        r0_start = 0
                    else:
                        r0_start = -self.rmax[0]
                    for r0 in range(r0_start, self.rmax[0]+1):
                        yield r0, r1, 0

        elif self.system.cell.nvec == 1:
            def rloops():
                for r0 in range(0, self.rmax[0]+1):
                    yield r0, 0, 0
        else:
            def rloops():
                yield 0, 0, 0

        # C) Compute the nlists the slow way
        validation = {}
        nvec = self.system.cell.nvec
        for r0, r1, r2 in rloops():
            for a in range(self.system.natom):
                for b in range(a+1):
                    if r0!=0 or r1!=0 or r2!=0:
                        signs = [1, -1]
                    elif a > b:
                        signs = [1]
                    else:
                        continue
                    for sign in signs:
                        delta = self.system.pos[b] - self.system.pos[a]
                        self.system.cell.mic(delta)
                        delta *= sign
                        if nvec > 0:
                            self.system.cell.add_vec(delta, np.array([r0, r1, r2])[:nvec])
                        d = np.linalg.norm(delta)
                        if d < self.rcut + self.skin:
                            if sign == 1:
                                key = a, b, r0, r1, r2
                            else:
                                key = b, a, r0, r1, r2
                            value = np.array([d, delta[0], delta[1], delta[2]])
                            validation[key] = value

        # D) Compare
        wrong = False
        with self.log.section('NLIST'):
            for key0, value0 in validation.items():
                value1 = actual.pop(key0, None)
                if value1 is None:
                    self.log('Missing:  ', key0)
                    self.log('  Validation %s %s %s %s' % (
                        self.log.length(value0[0]), self.log.length(value0[1]),
                        self.log.length(value0[2]), self.log.length(value0[3])
                    ))
                    wrong = True
                elif abs(value0 - value1).max() > 1e-10*self.log.length.conversion:
                    self.log('Different:', key0)
                    self.log('  Actual     %s %s %s %s' % (
                        self.log.length(value1[0]), self.log.length(value1[1]),
                        self.log.length(value1[2]), self.log.length(value1[3])
                    ))
                    self.log('  Validation %s %s %s %s' % (
                        self.log.length(value0[0]), self.log.length(value0[1]),
                        self.log.length(value0[2]), self.log.length(value0[3])
                    ))
                    self.log('  Difference %10.3e %10.3e %10.3e %10.3e' %
                        tuple((value0 - value1)/self.log.length.conversion)
                    )
                    self.log('  AbsMaxDiff %10.3e' %
                        (abs(value0 - value1).max()/self.log.length.conversion)
                    )
                    wrong = True
            for key1, value1 in actual.items():
                self.log('Redundant:', key1)
                self.log('  Actual     %s %s %s %s' % (
                    self.log.length(value1[0]), self.log.length(value1[1]),
                    self.log.length(value1[2]), self.log.length(value1[3])
                ))
                wrong = True
        assert not wrong


class FFArgs(object):
    '''Data structure that holds all arguments for the ForceField constructor
       The attributes of this object are gradually filled up by the various
       generators based on the data in the ParsedPars object.
    '''
    def __init__(self, log=None, timer=None, rcut=18.89726133921252, tr=Switch3(7.558904535685008),
                 alpha_scale=3.5, gcut_scale=1.1, skin=0, smooth_ei=False,
                 reci_ei='ewald', nlow=0, nhigh=-1, tailcorrections=False):
        """
           **Optional arguments:**
           Some optional arguments only make sense if related parameters in the
           parameter file are present.
           rcut
                The real space cutoff used by all pair potentials.
           tr
                Default truncation model for everything except the electrostatic
                interactions. The electrostatic interactions are not truncated
                by default.
           alpha_scale
                Determines the alpha parameter in the Ewald summation based on
                the real-space cutoff: alpha = alpha_scale / rcut. Higher
                values for this parameter imply a faster convergence of the
                reciprocal terms, but a slower convergence in real-space.
           gcut_scale
                Determines the reciprocale space cutoff based on the alpha
                parameter: gcut = gcut_scale * alpha. Higher values for this
                parameter imply a better convergence in the reciprocal space.
           skin
                The skin parameter for the neighborlist.
           smooth_ei
                Flag for smooth truncations for the electrostatic interactions.
           reci_ei
                The method to be used for the reciprocal contribution to the
                electrostatic interactions in the case of periodic systems. This
                must be one of 'ignore' or 'ewald' or 'ewald_interaction'.
                The options starting with 'ewald' are only supported for 3D
                periodic systems. If 'ewald_interaction' is chosen, the
                reciprocal contribution will not be included and it should be
                accounted for by using the :class:`EwaldReciprocalInteraction`
           nlow
                Interactions between atom pairs are only included if at least
                one atom index is higher than or equal to nlow. The default
                nlow=0 means no exclusion. Valence terms involving atoms with
                index lower than or equal to nlow will not be included.
           nhigh
                Interactions between atom pairs are only included if at least
                one atom index is smaller than nhigh. The default nhigh=-1
                means no exclusion. Valence terms involving atoms with index
                higher than nhigh will not be included.
                If nlow=nhigh, the system is divided into two parts and only
                pairs involving one atom of each part will be included. This is
                useful to calculate interaction energies in Monte Carlo
                simulations
           tailcorrections
                Boolean: if true, apply a correction for the truncation of the
                pair potentials assuming the system is homogeneous in the
                region where the truncation modifies the pair potential
           The actual value of gcut, which depends on both gcut_scale and
           alpha_scale, determines the computational cost of the reciprocal term
           in the Ewald summation. The default values are just examples. An
           optimal trade-off between accuracy and computational cost requires
           some tuning. Dimensionless scaling parameters are used to make sure
           that the numerical errors do not depend too much on the real space
           cutoff and the system size.
        """
        if reci_ei not in ['ignore', 'ewald', 'ewald_interaction']:
            raise ValueError('The reci_ei option must be one of \'ignore\' or \'ewald\' or \'ewald_interaction\'.')
        self.log = log
        self.timer = timer
        self.rcut = rcut
        self.tr = tr
        self.alpha_scale = alpha_scale
        self.gcut_scale = gcut_scale
        self.skin = skin
        self.smooth_ei = smooth_ei
        self.reci_ei = reci_ei
        # arguments for the ForceField constructor
        self.parts = []
        self.nlist = None
        self.nlow = nlow
        self.nhigh = nhigh
        self.tailcorrections = tailcorrections

    def get_nlist(self, system):
        if self.nlist is None:
            self.nlist = NeighborList(system, skin=self.skin, nlow=self.nlow,
                            nhigh=self.nhigh, log=self.log, timer=self.timer)
        return self.nlist

    def get_part(self, ForcePartClass):
        for part in self.parts:
            if isinstance(part, ForcePartClass):
                return part

    def get_part_pair(self, PairPotClass):
        for part in self.parts:
            if isinstance(part, ForcePartPair) and isinstance(part.pair_pot, PairPotClass):
                return part

    def get_part_valence(self, system):
        part_valence = self.get_part(ForcePartValence)
        if part_valence is None:
            part_valence = ForcePartValence(system, log=self.log, timer=self.timer)
            self.parts.append(part_valence)
        return part_valence

    def add_electrostatic_parts(self, system, scalings, dielectric):
        if self.get_part_pair(PairPotEI) is not None:
            return
        nlist = self.get_nlist(system)
        if system.cell.nvec == 0:
            alpha = 0.0
        elif system.cell.nvec == 3:
            #TODO: the choice of alpha should depend on the radii of the
            #charge distributions. Following expression is OK for point charges.
            alpha = self.alpha_scale/self.rcut
        else:
            raise NotImplementedError('Only zero- and three-dimensional electrostatics are supported.')
        # Real-space electrostatics
        if self.smooth_ei:
            pair_pot_ei = PairPotEI(system.charges, alpha, self.rcut, self.tr, dielectric, system.radii)
        else:
            pair_pot_ei = PairPotEI(system.charges, alpha, self.rcut, None, dielectric, system.radii)
        part_pair_ei = ForcePartPair(system, nlist, scalings, pair_pot_ei, log=self.log, timer=self.timer)
        self.parts.append(part_pair_ei)
        if self.reci_ei == 'ignore':
            # Nothing to do
            pass
        elif self.reci_ei.startswith('ewald'):
            if system.cell.nvec == 3:
                if self.reci_ei == 'ewald_interaction':
                    part_ewald_reci = ForcePartEwaldReciprocalInteraction(system.cell, alpha, self.gcut_scale*alpha, dielectric=dielectric, log=self.log, timer=self.timer)
                elif self.reci_ei == 'ewald':
                    # Reciprocal-space electrostatics
                    part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, self.gcut_scale*alpha, dielectric, self.nlow, self.nhigh, log=self.log, timer=self.timer)
                else: raise NotImplementedError
                self.parts.append(part_ewald_reci)
                # Ewald corrections
                part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings, dielectric, self.nlow, self.nhigh, log=self.log, timer=self.timer)
                self.parts.append(part_ewald_corr)
                # Neutralizing background
                part_ewald_neut = ForcePartEwaldNeutralizing(system, alpha, dielectric, self.nlow, self.nhigh, log=self.log, timer=self.timer)
                self.parts.append(part_ewald_neut)
            elif system.cell.nvec != 0:
                raise NotImplementedError('The ewald summation is only available for 3D periodic systems.')
        else:
            raise NotImplementedError

class FixedChargeGenerator(NonbondedGenerator):
    prefix = 'FIXQ'
    suffixes = ['UNIT', 'SCALE', 'ATOM', 'BOND', 'DIELECTRIC']
    par_info = [('Q0', float), ('P', float), ('R', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        atom_table = self.process_atoms(parsec['ATOM'], conversions)
        bond_table = self.process_bonds(parsec['BOND'], conversions)
        scale_table = self.process_scales(parsec['SCALE'])
        dielectric = self.process_dielectric(parsec['DIELECTRIC'])
        self.apply(atom_table, bond_table, scale_table, dielectric, system, ff_args)

    def process_atoms(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments')
            ffatype = words[0]
            if ffatype in result:
                pardef.complain(counter, 'has an atom type that was already encountered earlier')
            try:
                charge = float(words[1])*conversions['Q0']
                radius = float(words[2])*conversions['R']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to a floating point number')
            result[ffatype] = charge, radius
        return result

    def process_bonds(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments')
            key = tuple(words[:2])
            if key in result:
                pardef.complain(counter, 'has a combination of atom types that were already encountered earlier')
            try:
                charge_transfer = float(words[2])*conversions['P']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to floating point numbers')
            result[key] = charge_transfer
            result[key[::-1]] = -charge_transfer
        return result

    def process_dielectric(self, pardef):
        result = None
        for counter, line in pardef:
            if result is not None:
                pardef.complain(counter, 'is redundant. The DIELECTRIC suffix may only occur once')
            words = line.split()
            if len(words) != 1:
                pardef.complain(counter, 'must have one argument')
            try:
                result = float(words[0])
            except ValueError:
                pardef.complain(counter, 'must have a floating point argument')
        return result

    def apply(self, atom_table, bond_table, scale_table, dielectric, system, ff_args):
        if system.charges is None:
            system.charges = np.zeros(system.natom)
        elif ff_args.log.do_warning and abs(system.charges).max() != 0:
            ff_args.log.warn('Overwriting charges in system.')
        system.charges[:] = 0.0
        system.radii = np.zeros(system.natom)

        # compute the charges
        for i in range(system.natom):
            pars = atom_table.get(system.get_ffatype(i))
            if pars is not None:
                charge, radius = pars
                system.charges[i] += charge
                system.radii[i] = radius
            elif log.do_warning:
                ff_args.log.warn('No charge defined for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
        for i0, i1 in system.iter_bonds():
            ffatype0 = system.get_ffatype(i0)
            ffatype1 = system.get_ffatype(i1)
            if ffatype0 == ffatype1:
                continue
            charge_transfer = bond_table.get((ffatype0, ffatype1))
            if charge_transfer is None:
                if log.do_warning:
                    ff_args.log.warn('No charge transfer parameter for atom pair (%i,%i) with fftype (%s,%s).' % (i0, i1, system.get_ffatype(i0), system.get_ffatype(i1)))
            else:
                system.charges[i0] += charge_transfer
                system.charges[i1] -= charge_transfer

        # prepare other parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Setup the electrostatic pars
        ff_args.add_electrostatic_parts(system, scalings, dielectric)

class LJCrossGenerator(NonbondedGenerator):
    prefix = 'LJCROSS'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 2)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(par_table, scale_table, system, ff_args)

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def apply(self, par_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        nffatypes = system.ffatype_ids.max() + 1
        sigmas = np.zeros([nffatypes, nffatypes])
        epsilons = np.zeros([nffatypes, nffatypes])
        for i in range(system.natom):
            for j in range(system.natom):
                ffa_i, ffa_j = system.ffatype_ids[i], system.ffatype_ids[j]
                key = (system.get_ffatype(i), system.get_ffatype(j))
                par_list = par_table.get(key, [])
                if len(par_list) > 2:
                    raise TypeError('Superposition should not be allowed for non-covalent terms.')
                elif len(par_list) == 1:
                    sigmas[ffa_i,ffa_j], epsilons[ffa_i,ffa_j] = par_list[0]
                elif len(par_list) == 0:
                    if ff_args.log.do_high:
                        ff_args.log('No LJCross parameters found for ffatypes %s,%s. Parameters set to zero.' % (system.ffatypes[i0], system.ffatypes[i1]))

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotLJCross)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the LJCross part should not be present yet.')

        pair_pot = PairPotLJCross(system.ffatype_ids, epsilons, sigmas, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot, log=ff_args.log, timer=ff_args.timer)
        ff_args.parts.append(part_pair)

class MM3Generator(NonbondedGenerator):
    prefix = 'MM3'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float), ('ONLYPAULI', int)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(par_table, scale_table, system, ff_args)

    def apply(self, par_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        onlypaulis = np.zeros(system.natom, np.int32)
        for i in range(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) > 2:
                raise TypeError('Superposition should not be allowed for non-covalent terms.')
            elif len(par_list) == 1:
                sigmas[i], epsilons[i], onlypaulis[i] = par_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotMM3)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the MM3 part should not be present yet.')

        pair_pot = PairPotMM3(sigmas, epsilons, onlypaulis, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot, log=ff_args.log, timer=ff_args.timer)
        ff_args.parts.append(part_pair)


def apply_generators(system, parameters, ff_args):
    '''Populate the attributes of ff_args, prepares arguments for ForceField

       **Arguments:**

       system
            A System instance for which the force field object is being made

       ff_args
            An instance of the FFArgs class.

       parameters
            An instance of the Parameters, typically made by
            ``Parameters.from_file('parameters.txt')``.
    '''

    # Collect all the generators that have a prefix.
    import yaff.pes.generator as yaff_gen
    generators = {}

    for name, gen in yaff_gen.__dict__.items():
        if isinstance(gen, type) and issubclass(gen, Generator) and gen.prefix is not None:
            if name == 'FixedChargeGenerator':
                generators[FixedChargeGenerator.prefix] = FixedChargeGenerator()
            elif name == 'MM3Generator':
                generators[MM3Generator.prefix] = MM3Generator()
            elif name == 'LJCrossGenerator':
                generators[LJCrossGenerator.prefix] = LJCrossGenerator()
            else:
                generators[gen.prefix] = gen()

    # Go through all the sections of the parameter file and apply the
    # corresponding generator.
    for prefix, section in parameters.sections.items():
        generator = generators.get(prefix)
        if generator is None:
            if ffargs.log.do_warning:
                ffargs.log.warn('There is no generator named %s. It will be ignored.' % prefix)
        else:
            generator(system, section, ff_args)

    # If tail corrections are requested, go through all parts and add when necessary
    if ff_args.tailcorrections:
        if system.cell.nvec==0:
            ffargs.log.warn('Tail corrections were requested, but this makes no sense for non-periodic system. Not adding tail corrections...')
        elif system.cell.nvec==3:
            for part in ff_args.parts:
                # Only add tail correction to pair potentials
                if isinstance(part,ForcePartPair):
                    # Don't add tail corrections to electrostatic parts whose
                    # long-range interactions are treated using for instance Ewald
                    if isinstance(part.pair_pot,PairPotEI) or isinstance(part.pair_pot,PairPotEIDip):
                        continue
                    else:
                        part_tailcorrection = ForcePartTailCorrection(system, part, log = ff_args.log, timer = ff_args.timer)
                        ff_args.parts.append(part_tailcorrection)
        else:
            raise ValueError('Tail corrections not available for 1-D and 2-D periodic systems')

    part_valence = ff_args.get_part(ForcePartValence)
    if part_valence is not None and ff_args.log.do_warning:
        # Basic check for missing terms
        groups = set([])
        nv = part_valence.vlist.nv
        for iv in range(nv):
            # Get the atoms in the energy term.
            atoms = part_valence.vlist.lookup_atoms(iv)
            # Reduce it to a set of atom indices.
            atoms = frozenset(sum(sum(atoms, []), []))
            # Keep all two- and three-body terms.
            if len(atoms) <= 3:
                groups.add(atoms)
        # Check if some are missing
        for i0, i1 in system.iter_bonds():
            if frozenset([i0, i1]) not in groups:
                ff_args.log.warn('No covalent two-body term for atoms ({}, {})'.format(i0, i1))
        for i0, i1, i2 in system.iter_angles():
            if frozenset([i0, i1, i2]) not in groups:
                ff_args.log.warn('No covalent three-body term for atoms ({}, {} {})'.format(i0, i1, i2))


class ForcePart(object):
    '''Base class for anything that can compute energies (and optionally gradient
       and virial) for a ``System`` object.
    '''
    def __init__(self, name, system, log=None, timer=None):
        """
           **Arguments:**

           name
                A name for this part of the force field. This name must adhere
                to the following conventions: all lower case, no white space,
                and short. It is used to construct part_* attributes in the
                ForceField class, where * is the name.

           system
                The system to which this part of the FF applies.
        """
        self.name = name
        # backup copies of last call to compute:
        self.energy = 0.0
        self.gpos = np.zeros((system.natom, 3), float)
        self.vtens = np.zeros((3, 3), float)
        self.log = log
        self.timer = timer
        self.clear()

    def clear(self):
        """Fill in nan values in the cached results to indicate that they have
           become invalid.
        """
        self.energy = np.nan
        self.gpos[:] = np.nan
        self.vtens[:] = np.nan

    def update_rvecs(self, rvecs):
        '''Let the ``ForcePart`` object know that the cell vectors have changed.

           **Arguments:**

           rvecs
                The new cell vectors.
        '''
        self.clear()

    def update_pos(self, pos):
        '''Let the ``ForcePart`` object know that the atomic positions have changed.

           **Arguments:**

           pos
                The new atomic coordinates.
        '''
        self.clear()

    def compute(self, gpos=None, vtens=None):
        """Compute the energy and optionally some derivatives for this FF (part)

           The only variable inputs for the compute routine are the atomic
           positions and the cell vectors, which can be changed through the
           ``update_rvecs`` and ``update_pos`` methods. All other aspects of
           a force field are considered to be fixed between subsequent compute
           calls. If changes other than positions or cell vectors are needed,
           one must construct new ``ForceField`` and/or ``ForcePart`` objects.

           **Optional arguments:**

           gpos
                The derivatives of the energy towards the Cartesian coordinates
                of the atoms. ('g' stands for gradient and 'pos' for positions.)
                This must be a writeable numpy array with shape (N, 3) where N
                is the number of atoms.

           vtens
                The force contribution to the pressure tensor. This is also
                known as the virial tensor. It represents the derivative of the
                energy towards uniform deformations, including changes in the
                shape of the unit cell. (v stands for virial and 'tens' stands
                for tensor.) This must be a writeable numpy array with shape (3,
                3).

           The energy is returned. The optional arguments are Fortran-style
           output arguments. When they are present, the corresponding results
           are computed and **added** to the current contents of the array.
        """
        if gpos is None:
            my_gpos = None
        else:
            my_gpos = self.gpos
            my_gpos[:] = 0.0
        if vtens is None:
            my_vtens = None
        else:
            my_vtens = self.vtens
            my_vtens[:] = 0.0
        self.energy = self._internal_compute(my_gpos, my_vtens)
        if np.isnan(self.energy):
            raise ValueError('The energy is not-a-number (nan).')
        if gpos is not None:
            if np.isnan(my_gpos).any():
                if self.name:
                    raise ValueError('Some gpos element(s) is/are not-a-number (nan) for {}'.format(self.name))
                raise ValueError('Some gpos element(s) is/are not-a-number (nan).')
            gpos += my_gpos
        if vtens is not None:
            if np.isnan(my_vtens).any():
                raise ValueError('Some vtens element(s) is/are not-a-number (nan).')
            vtens += my_vtens
        return self.energy

    def _internal_compute(self, gpos, vtens):
        '''Subclasses implement their compute code here.'''
        raise NotImplementedError


class ForceField(ForcePart):
    '''A complete force field model.'''
    def __init__(self, system, parts, nlist=None, log=None, timer=None):
        """
           **Arguments:**

           system
                An instance of the ``System`` class.

           parts
                A list of instances of sublcasses of ``ForcePart``. These are
                the different types of contributions to the force field, e.g.
                valence interactions, real-space electrostatics, and so on.

           **Optional arguments:**

           nlist
                A ``NeighborList`` instance. This is required if some items in the
                parts list use this nlist object.
        """
        ForcePart.__init__(self, 'all', system, log=log, timer=timer)
        self.system = system
        self.parts = []
        self.nlist = nlist
        self.needs_nlist_update = nlist is not None
        for part in parts:
            self.add_part(part)
        if self.log.do_medium:
            with self.log.section('FFINIT'):
                self.log('Force field with %i parts:&%s.' % (
                    len(self.parts), ', '.join(part.name for part in self.parts)
                ))
                self.log('Neighborlist present: %s' % (self.nlist is not None))

    def add_part(self, part):
        # Check if all parts have a log attribute
        if not hasattr(part,'log'):
            print(part.name, 'has no log attribute.')
            part.log = self.log
            part.timer = self.timer
        self.parts.append(part)
        # Make the parts also accessible as simple attributes.
        name = 'part_%s' % part.name
        if name in self.__dict__:
            raise ValueError('The part %s occurs twice in the force field.' % name)
        self.__dict__[name] = part

    @classmethod
    def generate(cls, system, parameters, log=None, timer=None, **kwargs):
        """Create a force field for the given system with the given parameters.

           **Arguments:**

           system
                An instance of the System class

           parameters
                Three types are accepted: (i) the filename of the parameter
                file, which is a text file that adheres to YAFF parameter
                format, (ii) a list of such filenames, or (iii) an instance of
                the Parameters class.

           See the constructor of the :class:`yaff.pes.generator.FFArgs` class
           for the available optional arguments.

           This method takes care of setting up the FF object, and configuring
           all the necessary FF parts. This is a lot easier than creating an FF
           with the default constructor. Parameters for atom types that are not
           present in the system, are simply ignored.
        """
        cls_log = log
        cls_timer = timer
        if system.ffatype_ids is None:
            raise ValueError('The generators needs ffatype_ids in the system object.')
        with cls_log.section('GEN'), cls_timer.section('Generator'):
            from yaff.pes.parameters import Parameters
            if cls_log.do_medium:
                cls_log('Generating force field from %s' % str(parameters))
            if not isinstance(parameters, Parameters):
                parameters = Parameters.from_file(parameters)
            ff_args = FFArgs(log=cls_log,timer=cls_timer,**kwargs)
            apply_generators(system, parameters, ff_args)
            return ForceField(system, ff_args.parts, ff_args.nlist, log=cls_log, timer=cls_timer)

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.system.cell.update_rvecs(rvecs)
        if self.nlist is not None:
            self.nlist.update_rmax()
            self.needs_nlist_update = True

    def update_pos(self, pos):
        '''See :meth:`yaff.pes.ff.ForcePart.update_pos`'''
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos
        if self.nlist is not None:
            self.needs_nlist_update = True

    def _internal_compute(self, gpos, vtens):
        if self.needs_nlist_update:
            self.nlist.update()
            self.needs_nlist_update = False
        result = sum([part.compute(gpos, vtens) for part in self.parts])
        return result


class ForcePartPair(ForcePart):
    '''A pairwise (short-range) non-bonding interaction term.

       This part can be used for the short-range electrostatics, Van der Waals
       terms, etc. Currently, one has to use multiple ``ForcePartPair``
       objects in a ``ForceField`` in order to combine different types of pairwise
       energy terms, e.g. to combine an electrostatic term with a Van der
       Waals term. (This may be changed in future to improve the computational
       efficiency.)
    '''
    def __init__(self, system, nlist, scalings, pair_pot, log=None, timer=None):
        '''
           **Arguments:**

           system
                The system to which this pairwise interaction applies.

           nlist
                A ``NeighborList`` object. This has to be the same as the one
                passed to the ForceField object that contains this part.

           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.

           pair_pot
                An instance of the ``PairPot`` built-in class from
                :mod:`yaff.pes.ext`.
        '''
        ForcePart.__init__(self, 'pair_%s' % pair_pot.name, system, log=log, timer=timer)
        self.nlist = nlist
        self.scalings = scalings
        self.pair_pot = pair_pot
        self.nlist.request_rcut(pair_pot.rcut)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                self.log('  real space cutoff: %s' % self.log.length(pair_pot.rcut))
                tr = pair_pot.get_truncation()
                if tr is None:
                    self.log('  truncation:     none')
                else:
                    self.log('  truncation:     %s' % tr.get_log())
                if self.log.do_medium and isinstance(self.pair_pot, PairPotEI):
                    self.log('  alpha:                 %s' % log.invlength(self.pair_pot.alpha))
                    self.log('  relative permittivity: %5.3f' % self.pair_pot.dielectric)
                else:
                    self.pair_pot.log() # Screen Logging in C-code
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('PP %s' % self.pair_pot.name):
            return self.pair_pot.compute(self.nlist.neighs, self.scalings.stab, gpos, vtens, self.nlist.nneigh)


class ForcePartValence(ForcePart):
    '''The covalent part of a force-field model.

       The covalent force field is implemented in a three-layer approach,
       similar to the implementation of a neural network:

       (0. Optional, not used by default. A layer that computes centers of mass for groups
           of atoms.)

       1. The first layer consists of a :class:`yaff.pes.dlist.DeltaList` object
          that computes all the relative vectors needed for the internal
          coordinates in the covalent energy terms. This list is automatically
          built up as energy terms are added with the ``add_term`` method. This
          list also takes care of transforming `derivatives of the energy
          towards relative vectors` into `derivatives of the energy towards
          Cartesian coordinates and the virial tensor`.

       2. The second layer consist of a
          :class:`yaff.pes.iclist.InternalCoordinateList` object that computes
          the internal coordinates, based on the ``DeltaList``. This list is
          also automatically built up as energy terms are added. The same list
          is also responsible for transforming `derivatives of the energy
          towards internal coordinates` into `derivatives of the energy towards
          relative vectors`.

       3. The third layers consists of a :class:`yaff.pes.vlist.ValenceList`
          object. This list computes the covalent energy terms, based on the
          result in the ``InternalCoordinateList``. This list also computes the
          derivatives of the energy terms towards the internal coordinates.

       The computation of the covalent energy is the so-called `forward code
       path`, which consists of running through steps 1, 2 and 3, in that order.
       The derivatives of the energy are computed in the so-called `backward
       code path`, which consists of taking steps 1, 2 and 3 in reverse order.
       This basic idea of back-propagation for the computation of derivatives
       comes from the field of neural networks. More details can be found in the
       chapter, :ref:`dg_sec_backprop`.
    '''
    def __init__(self, system, comlist=None, log=None, timer=None):
        '''
           Parameters
           ----------

           system
                An instance of the ``System`` class.
           comlist
                An optional layer to derive centers of mass from the atomic positions.
                These centers of mass are used as input for the first layer, the relative
                vectors.
        '''
        ForcePart.__init__(self, 'valence', system, log=log, timer=timer)
        self.comlist = comlist
        self.dlist = DeltaList(system if comlist is None else comlist)
        self.iclist = InternalCoordinateList(self.dlist)
        self.vlist = ValenceList(self.iclist)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def add_term(self, term):
        '''Add a new term to the covalent force field.

           **Arguments:**

           term
                An instance of the class :class:`yaff.pes.ff.vlist.ValenceTerm`.

           In principle, one should add all energy terms before calling the
           ``compute`` method, but with the current implementation of Yaff,
           energy terms can be added at any time. (This may change in future.)
        '''
        if self.log.do_high:
            with self.log.section('VTERM'):
                self.log('%7i&%s %s' % (self.vlist.nv, term.get_log(), ' '.join(ic.get_log() for ic in term.ics)))
        self.vlist.add_term(term)

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Valence'):
            if self.comlist is not None:
                self.comlist.forward()
            self.dlist.forward()
            self.iclist.forward()
            energy = self.vlist.forward()
            if not ((gpos is None) and (vtens is None)):
                self.vlist.back()
                self.iclist.back()
                if self.comlist is None:
                    self.dlist.back(gpos, vtens)
                else:
                    self.comlist.gpos[:] = 0.0
                    self.dlist.back(self.comlist.gpos, vtens)
                    self.comlist.back(gpos, vtens)
            return energy


class ForcePartPressure(ForcePart):
    '''Applies a constant istropic pressure.'''
    def __init__(self, system, pext, log=None, timer=None):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           pext
                The external pressure. (Positive will shrink the system.) In
                case of 2D-PBC, this is the surface tension. In case of 1D, this
                is the linear strain.

           This force part is only applicable to systems that are periodic.
        '''
        if system.cell.nvec == 0:
            raise ValueError('The system must be periodic in order to apply a pressure')
        ForcePart.__init__(self, 'press', system, log=log, timer=timer)
        self.system = system
        self.pext = pext
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Valence'):
            cell = self.system.cell
            if (vtens is not None):
                rvecs = cell.rvecs
                if cell.nvec == 1:
                    vtens += self.pext/cell.volume*np.outer(rvecs[0], rvecs[0])
                elif cell.nvec == 2:
                    vtens += self.pext/cell.volume*(
                          np.dot(rvecs[0], rvecs[0])*np.outer(rvecs[1], rvecs[1])
                        + np.dot(rvecs[0], rvecs[0])*np.outer(rvecs[1], rvecs[1])
                        - np.dot(rvecs[1], rvecs[0])*np.outer(rvecs[0], rvecs[1])
                        - np.dot(rvecs[0], rvecs[1])*np.outer(rvecs[1], rvecs[0])
                    )
                elif cell.nvec == 3:
                    gvecs = cell.gvecs
                    vtens += self.pext*cell.volume*np.identity(3)
                else:
                    raise NotImplementedError
            return cell.volume*self.pext


class ForcePartGrid(ForcePart):
    '''Energies obtained by grid interpolation.'''
    def __init__(self, system, grids, log=None, timer=None):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           grids
                A dictionary with (ffatype, grid) items. Each grid must be a
                three-dimensional array with energies.

           This force part is only applicable to systems that are 3D periodic.
        '''
        if system.cell.nvec != 3:
            raise ValueError('The system must be 3d periodic for the grid term.')
        for grid in grids.values():
            if grid.ndim != 3:
                raise ValueError('The energy grids must be 3D numpy arrays.')
        ForcePart.__init__(self, 'grid', system, log=log, timer=timer)
        self.system = system
        self.grids = grids
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Grid'):
            if gpos is not None:
                raise NotImplementedError('Cartesian gradients are not supported yet in ForcePartGrid')
            if vtens is not None:
                raise NotImplementedError('Cell deformation are not supported by ForcePartGrid')
            cell = self.system.cell
            result = 0
            for i in range(self.system.natom):
                grid = self.grids[self.system.get_ffatype(i)]
                result += compute_grid3d(self.system.pos[i], cell, grid)
            return result


class ForcePartTailCorrection(ForcePart):
    '''Corrections to energy and virial tensor to compensate for neglecting
    pair potentials at long range'''
    def __init__(self, system, part_pair, log=None, timer=None):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           part_pair
                An instance of the ``PairPot`` class.

           This force part is only applicable to systems that are 3D periodic.
        '''
        if system.cell.nvec != 3:
            raise ValueError('Tail corrections can only be applied to 3D periodic systems')
        if part_pair.name in ['pair_ei','pair_eidip']:
            raise ValueError('Tail corrections are divergent for %s'%part_pair.name)
        super(ForcePartTailCorrection, self).__init__('tailcorr_%s'%(part_pair.name), system, log=log, timer=timer)
        self.ecorr, self.wcorr = part_pair.pair_pot.prepare_tailcorrections(system.natom)
        self.system = system
        self.log = log
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        if vtens is not None:
            w = 2.0*np.pi*self.wcorr/self.system.cell.volume
            vtens[0,0] += w
            vtens[1,1] += w
            vtens[2,2] += w
        return 2.0*np.pi*self.ecorr/self.system.cell.volume


class ForcePartEwaldReciprocal(ForcePart):
    '''The long-range contribution to the electrostatic interaction in 3D
       periodic systems.
    '''
    def __init__(self, system, alpha, gcut=0.35, dielectric=1.0, nlow=0, nhigh=-1, log=None, timer=None):
        '''
           **Arguments:**
           system
                The system to which this interaction applies.
           alpha
                The alpha parameter in the Ewald summation method.
           **Optional arguments:**
           gcut
                The cutoff in reciprocal space.
           dielectric
                The scalar relative permittivity of the system.
           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.
           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion..
        '''
        ForcePart.__init__(self, 'ewald_reci', system, log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.gcut = gcut
        self.dielectric = dielectric
        self.update_gmax()
        self.work = np.empty(system.natom*2)
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:                 %s' % self.log.invlength(self.alpha))
                self.log('  gcut:                  %s' % self.log.invlength(self.gcut))
                self.log('  relative permittivity: %5.3f' % self.dielectric)
                self.log.hline()


    def update_gmax(self):
        '''This routine must be called after the attribute self.gmax is modified.'''
        self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)
        if self.log.do_debug:
            with self.log.section('EWALD'):
                self.log('gmax a,b,c   = %i,%i,%i' % tuple(self.gmax))

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.update_gmax()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald reci.'):
            return compute_ewald_reci(
                self.system.pos, self.system.charges, self.system.cell, self.alpha,
                self.gmax, self.gcut, self.dielectric, gpos, self.work, vtens, self.nlow, self.nhigh
            )


class ForcePartEwaldReciprocalDD(ForcePart):
    '''The long-range contribution to the dipole-dipole
       electrostatic interaction in 3D periodic systems.
    '''
    def __init__(self, system, alpha, gcut=0.35, nlow=0, nhigh=-1, log=None, timer=None):
        '''
           **Arguments:**
           system
                The system to which this interaction applies.
           alpha
                The alpha parameter in the Ewald summation method.
           gcut
                The cutoff in reciprocal space.
           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.
           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
        '''
        ForcePart.__init__(self, 'ewald_reci', system, log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        if system.dipoles is None:
            raise ValueError('The system does not have dipoles.')
        self.system = system
        self.alpha = alpha
        self.gcut = gcut
        self.update_gmax()
        self.work = np.empty(system.natom*2)
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:             %s' % self.log.invlength(self.alpha))
                self.log('  gcut:              %s' % self.log.invlength(self.gcut))
                self.log.hline()


    def update_gmax(self):
        '''This routine must be called after the attribute self.gmax is modified.'''
        self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)
        if self.log.do_debug:
            with self.log.section('EWALD'):
                self.log('gmax a,b,c   = %i,%i,%i' % tuple(self.gmax))

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.update_gmax()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald reci.'):
            return compute_ewald_reci_dd(
                self.system.pos, self.system.charges, self.system.dipoles, self.system.cell, self.alpha,
                self.gmax, self.gcut, gpos, self.work, vtens, self.nlow, self.nhigh
            )


class ForcePartEwaldCorrection(ForcePart):
    '''Correction for the double counting in the long-range term of the Ewald sum.
       This correction is only needed if scaling rules apply to the short-range
       electrostatics.
    '''
    def __init__(self, system, alpha, scalings, dielectric=1.0, nlow=0, nhigh=-1, log=None, timer=None):
        '''
           **Arguments:**
           system
                The system to which this interaction applies.
           alpha
                The alpha parameter in the Ewald summation method.
           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.
           **Optional arguments:**
           dielectric
                The scalar relative permittivity of the system.
           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.
           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
        '''
        ForcePart.__init__(self, 'ewald_cor', system, log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.dielectric = dielectric
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        self.scalings = scalings
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:             %s' % self.log.invlength(self.alpha))
                self.log('  relative permittivity   %5.3f' % self.dielectric)
                self.log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald corr.'):
            return compute_ewald_corr(
                self.system.pos, self.system.charges, self.system.cell,
                self.alpha, self.scalings.stab, self.dielectric, gpos, vtens, self.nlow, self.nhigh
            )


class ForcePartEwaldCorrectionDD(ForcePart):
    '''Correction for the double counting in the long-range term of the Ewald sum.
       This correction is only needed if scaling rules apply to the short-range
       electrostatics.
    '''
    def __init__(self, system, alpha, scalings, nlow=0, nhigh=-1, log=None, timer=None):
        '''
           **Arguments:**
           system
                The system to which this interaction applies.
           alpha
                The alpha parameter in the Ewald summation method.
           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.
           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.
           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
        '''
        ForcePart.__init__(self, 'ewald_cor', system, log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        self.scalings = scalings
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:             %s' % self.log.invlength(self.alpha))
                self.log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald corr.'):
            return compute_ewald_corr_dd(
                self.system.pos, self.system.charges, self.system.dipoles, self.system.cell,
                self.alpha, self.scalings.stab, gpos, vtens, self.nlow, self.nhigh
            )

class ForcePartEwaldNeutralizing(ForcePart):
    '''Neutralizing background correction for 3D periodic systems that are
       charged.
       This term is only required of the system is not neutral.
    '''
    def __init__(self, system, alpha, dielectric=1.0, nlow=0, nhigh=-1,
                fluctuating_charges=False, log=None, timer=None):
        '''
           **Arguments:**
           system
                The system to which this interaction applies.
           alpha
                The alpha parameter in the Ewald summation method.
           **Optional arguments:**
           dielectric
                The scalar relative permittivity of the system.
           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.
           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
           fluctuating_charges
                Boolean indicating whether charges (and radii) are allowed to
                change during a simulation. If set to False, some factors can
                be precomputed at the start of the simulation.
        '''
        ForcePart.__init__(self, 'ewald_neut', system, log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.dielectric = dielectric
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        self.fluctuating_charges = fluctuating_charges
        if not self.fluctuating_charges:
            fac = self.system.charges[:].sum()**2/self.alpha**2
            fac -= self.system.charges[:self.nlow].sum()**2/self.alpha**2
            fac -= self.system.charges[self.nhigh:].sum()**2/self.alpha**2
            if self.system.radii is not None:
                fac -= self.system.charges.sum()*np.sum( self.system.charges*self.system.radii**2 )
                fac += self.system.charges[:self.nlow].sum()*np.sum( self.system.charges[:self.nlow]*self.system.radii[:self.nlow]**2)
                fac += self.system.charges[self.nhigh:].sum()*np.sum( self.system.charges[self.nhigh:]*self.system.radii[self.nhigh:]**2)
            self.prefactor = fac*np.pi/(2.0*self.dielectric)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:                   %s' % self.log.invlength(self.alpha))
                self.log('  relative permittivity:   %5.3f' % self.dielectric)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald neut.'):
            if not self.fluctuating_charges:
                fac = self.prefactor/self.system.cell.volume
            else:
                #TODO: interaction of dipoles with background? I think this is zero, need proof...
                fac = self.system.charges[:].sum()**2/self.alpha**2
                fac -= self.system.charges[:self.nlow].sum()**2/self.alpha**2
                fac -= self.system.charges[self.nhigh:].sum()**2/self.alpha**2
                if self.system.radii is not None:
                    fac -= self.system.charges.sum()*np.sum( self.system.charges*self.system.radii**2 )
                    fac += self.system.charges[:self.nlow].sum()*np.sum( self.system.charges[:self.nlow]*self.system.radii[:self.nlow]**2)
                    fac += self.system.charges[self.nhigh:].sum()*np.sum( self.system.charges[self.nhigh:]*self.system.radii[self.nhigh:]**2)
                fac *= np.pi/(2.0*self.system.cell.volume*self.dielectric)
            if vtens is not None:
                vtens.ravel()[::4] -= fac
        return fac

class ForcePartEwaldReciprocalInteraction(ForcePart):
    r'''The reciprocal part of the Ewald summation, not the entire energy but
       only interactions between parts of the system. This allows a
       computationally very efficient evaluation of the energy difference when
       a limited number of atoms are moved, and is thus mostly useful in MC
       simulations. Although it is technically a subclass of ForcePart, it will
       not actually contribute to a ForceField. Because this class has to
       flexibility to handle varying numbers of atoms, it is only useful through
       direct calls of `compute_deltae`
       The reciprocal part of the Ewald summation for a set of N atoms is given
       by:
       .. math:: E = \frac{4\pi}{V} \sum_{\mathbf{k}} |S(\mathbf{k})|^2 \
                 \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2} \\
       where the so-called structure factors are given:
       .. math:: S(\mathbf{k}) = \sum_{i=1}^{N} q_i \
                 e^{j\mathbf{k}\cdot\mathbf{r}_i}
       Suppose that we want to compute the interaction energy with a set of
       M other atoms. This can be done as follows:
       .. math:: \Delta S(\mathbf{k}) = \sum_{i=1}^{M} q_i \
                 e^{j\mathbf{k}\cdot\mathbf{r}_i}
       Using the change in structure factors, the corresponding energy change
       is given by:
       .. math:: E = \frac{4\pi}{V} \sum_{\mathbf{k}} \left[ \bar{S}\Delta S \
                 + S\bar{\Delta S} \right] \
                 \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2} \\
       The structure factors are stored as an attribute of this Class. This
       makes it easy to add the contribution from new atoms to the existing
       structure factors, and in this way allowing tho handle a varying number
       of atoms.
       Only insertions are supported here, but deletions can be achieved by
       taking the negative of the change in structure factors. Translations and
       rotations can be achieved by combining a deletion and an insertion.
       Note that flexible cells are NOT supported => TODO check this
    '''
    def __init__(self, cell, alpha, gcut, pos=None, charges=None, dielectric=1.0, log=None, timer=None):
        '''
            **Arguments:**
            cell
                An instance of the ``Cell`` class.
            alpha
                The alpha parameter in the Ewald summation method.
            gcut
                The cutoff in reciprocal space.
            **Optional arguments:**
            pos
                A [Nx3] Numpy array, providing the coordinates of the atoms
                that are originally present.
            charges
                A [N] Numpy array, providing the charges of the atoms that are
                originally present.
           dielectric
                The scalar relative permittivity of the system.
        '''
        # Dummy attributes to keep things consistent with ForcePart,
        # these are not actually used.
        self.log = log
        self.timer = timer
        self.name = 'ewald_reciprocal_interaction'
        self.energy = 0.0
        self.gpos = np.zeros((0, 3), float)
        self.vtens = np.zeros((3, 3), float)
        if not cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        self.cell = cell
        # Store the original rvecs. If these would change, we need to
        # reinitialize
        self.rvecs0 = self.cell.rvecs.copy()
        self.alpha = alpha
        self.gcut = gcut
        self.dielectric = dielectric
        self.initialize()
        # Compute the structure factors if an initial configuration is
        # provided.
        if pos is not None:
            assert charges is not None
            self.compute_structurefactors(pos, charges, self.cosfacs, self.sinfacs)
        if self.log.do_medium:
            with self.log.section('EWIINIT'):
                self.log('Ewald Reciprocal interactions')
                self.log.hline()
                self.log('  alpha:             %s' % self.log.invlength(self.alpha))
                self.log('  gcut:              %s' % self.log.invlength(self.gcut))
                self.log.hline()

    def update_gmax(self):
        '''This routine must be called after the attribute self.gmax is modified.'''
        self.gmax = np.ceil(self.gcut/self.cell.gspacings-0.5).astype(int)
        if self.log.do_debug:
            with self.log.section('EWALDI'):
                self.log('gmax a,b,c   = %i,%i,%i' % tuple(self.gmax))

    def initialize(self):
        # Prepare the prefactors \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2}
        self.update_gmax()
        self.prefactors = np.zeros((2*self.gmax[0]+1,2*self.gmax[1]+1,self.gmax[2]+1))
        compute_ewald_prefactors(self.cell, self.alpha, self.gmax, self.gcut,
                self.prefactors)
        # Prepare the structure factors
        self.cosfacs = np.zeros(self.prefactors.shape)
        self.sinfacs = np.zeros(self.prefactors.shape)
        self.rvecs0 = self.cell.rvecs.copy()

    def compute_structurefactors(self, pos, charges, cosfacs, sinfacs):
        '''Compute the structure factors
           .. math:: \Delta S(\mathbf{k}) = \sum_{i=1}^{M} q_i \
                 e^{j\mathbf{k}\cdot\mathbf{r}_i}
           for the given coordinates and charges. The resulting real part is
           ADDED to cosfacs, the resulting imaginary part is ADDED to sinfacs.
        '''
        with self.timer.section('Ew.reci.SF'):
            if not np.all(self.cell.rvecs==self.rvecs0):
                if self.log.do_medium:
                    with self.log.section('EWALDI'):
                        self.log('Cell change detected, reinitializing')
                self.initialize()
            compute_ewald_structurefactors(pos, charges, self.cell, self.alpha,
                self.gmax, self.gcut, cosfacs, sinfacs)

    def compute_deltae(self, cosfacs, sinfacs):
        '''Compute the energy difference arising if the provided structure
           factors would be added to the current structure factors
        '''
        with self.timer.section('Ew.reci.int.'):
            e = compute_ewald_deltae(self.prefactors, cosfacs, self.cosfacs,
                 sinfacs, self.sinfacs)
        return e/self.dielectric

    def insertion_energy(self, pos, charges, cosfacs=None, sinfacs=None, sign=1):
        '''
        Compute the energy difference if atoms with given coordinates and
        charges are added to the systems. By setting sign to -1, the energy
        difference for removal of those atoms is returned.
            **Arguments:**
            pos
                [Nx3] NumPy array specifying the coordinates
            charges
                [N] NumPy array speficying the charges
            **Optional arguments:**
            cosfacs
                NumPy array with the same shape as the prefactors.
                If not provided, a new array will be created.
                If provided, existing entries will be zerod at the start,
                and contain cosine structure factors of the atoms at the end.
            sinfacs
                NumPy array with the same shape as the prefactors.
                If not provided, a new array will be created.
                If provided, existing entries will be zerod at the start.
                and contain sine structure factors of the atoms at the end.
            sign
                When set to 1, insertion is considered.
                When set to -1, deletion is considered.
        '''
        assert sign in [-1,1]
        if cosfacs is None:
            assert sinfacs is None
            cosfacs = np.zeros(self.prefactors.shape)
            sinfacs = np.zeros(self.prefactors.shape)
        else:
            cosfacs[:] = 0.0
            sinfacs[:] = 0.0
        self.compute_structurefactors(
                pos, charges, cosfacs, sinfacs)
        # We consider a deletion; this means that the structure factors
        # of the considered atoms need to be subtracted from the
        # current system structure factors
        if sign==-1:
            self.cosfacs[:] -= cosfacs
            self.sinfacs[:] -= sinfacs
        return sign*self.compute_deltae(cosfacs, sinfacs)

    def _internal_compute(self, gpos, vtens):
        return 0.0


def check_nlow_nhigh(system, nlow, nhigh):
    if nlow < 0:
        raise ValueError('nlow must be positive.')
    if nlow > system.natom:
        raise ValueError('nlow must not be larger than system.natom')
    if nhigh == -1: nhigh = system.natom
    if nhigh < nlow:
        raise ValueError('nhigh must not be smaller than nlow')
    if nhigh > system.natom:
        raise ValueError('nhigh must not be larger than system.natom')
    return nlow, nhigh

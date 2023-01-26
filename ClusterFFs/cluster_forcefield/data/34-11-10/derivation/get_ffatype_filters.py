#! /usr/bin/env python

from molmod.molecular_graphs import MolecularGraph
from molmod.molecular_graphs import HasAtomNumber, HasNumNeighbors, HasNeighborNumbers, HasNeighbors
from molmod.graphs import CritAnd, CritNot, CritOr

N = HasAtomNumber(7)
C = CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 7, 7))

C1_term = CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 6))
C2_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C1_term))
C3_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C2_term))
C4_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C3_term))
H3_term = CritAnd(HasAtomNumber(1), HasNeighbors(C2_term))
H4_term = CritAnd(HasAtomNumber(1), HasNeighbors(C3_term))
H5_term = CritAnd(HasAtomNumber(1), HasNeighbors(C4_term))

afilters = [('C_34-11-34', C), ('N_34-11-34', N), ('C1_term', C1_term), ('C2_term', C2_term), ('C3_term', C3_term), ('C4_term', C4_term), ('H3_term', H3_term), ('H4_term', H4_term), ('H5_term', H5_term)]


#! /usr/bin/env python

from molmod.molecular_graphs import MolecularGraph
from molmod.molecular_graphs import HasAtomNumber, HasNumNeighbors, HasNeighborNumbers, HasNeighbors
from molmod.graphs import CritAnd, CritNot, CritOr

B_HN2_B2C2 = CritAnd(HasAtomNumber(5))
N_B2C_H2C2N2 = CritAnd(HasAtomNumber(7))
H_B_N2 = CritAnd(HasAtomNumber(1), HasNeighborNumbers(5))

C1_term = CritAnd(HasAtomNumber(6), HasNeighborNumbers(7, 6, 6))
C2_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C1_term))
C3_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C2_term))
C4_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C3_term))
H3_term = CritAnd(HasAtomNumber(1), HasNeighbors(C2_term))
H4_term = CritAnd(HasAtomNumber(1), HasNeighbors(C3_term))
H5_term = CritAnd(HasAtomNumber(1), HasNeighbors(C4_term))

afilters = [('B_36-11-36', B_HN2_B2C2), ('N_36-11-36', N_B2C_H2C2N2), ('H_36-11-16', H_B_N2), ('C1_term', C1_term), ('C2_term', C2_term), ('C3_term', C3_term), ('C4_term', C4_term), ('H3_term', H3_term), ('H4_term', H4_term), ('H5_term', H5_term)]


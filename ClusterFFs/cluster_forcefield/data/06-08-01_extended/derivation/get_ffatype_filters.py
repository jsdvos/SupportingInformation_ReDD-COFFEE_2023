#! /usr/bin/env python

from molmod.molecular_graphs import MolecularGraph
from molmod.molecular_graphs import HasAtomNumber, HasNumNeighbors, HasNeighborNumbers, HasNeighbors
from molmod.graphs import CritAnd, CritNot, CritOr

O_THB = CritAnd(HasAtomNumber(8), HasNeighborNumbers(5, 6))
B_THB = CritAnd(HasAtomNumber(5), HasNeighbors(HasAtomNumber(6), O_THB, O_THB))
C_O_THB = CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 8))
C_O_BR_THB = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), C_O_THB, C_O_THB))
H_O_BR_THB = CritAnd(HasAtomNumber(1), HasNeighbors(C_O_BR_THB))

C1_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), HasAtomNumber(6), B_THB))
C2_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C1_term))
C3_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(5), HasAtomNumber(6), C2_term))
C4_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), C3_term, C3_term))
H3_term = CritAnd(HasAtomNumber(1), HasNeighbors(C2_term))
H5_term = CritAnd(HasAtomNumber(1), HasNeighbors(C4_term))
B4_term = CritAnd(HasAtomNumber(5), HasNeighbors(HasAtomNumber(8), HasAtomNumber(8), C3_term))
O5_term = CritAnd(HasAtomNumber(8), HasNeighbors(HasAtomNumber(1), B4_term))
H6_term = CritAnd(HasAtomNumber(1), HasNeighbors(O5_term))

afilters = [('B_THB', B_THB), ('O_THB', O_THB), ('C_O_THB', C_O_THB), ('C_O_BR_THB', C_O_BR_THB), ('H_O_BR_THB', H_O_BR_THB), ('C1_term', C1_term), ('C2_term', C2_term), ('C3_term', C3_term), ('C4_term', C4_term), ('H3_term', H3_term), ('H5_term', H5_term), ('B4_term', B4_term), ('O5_term', O5_term), ('H6_term', H6_term)]



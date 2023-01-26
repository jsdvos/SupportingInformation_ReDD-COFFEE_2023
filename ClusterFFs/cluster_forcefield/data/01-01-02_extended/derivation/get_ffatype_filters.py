#! /usr/bin/env python

from molmod.molecular_graphs import MolecularGraph
from molmod.molecular_graphs import HasAtomNumber, HasNumNeighbors, HasNeighborNumbers, HasNeighbors
from molmod.graphs import CritAnd, CritNot, CritOr

C_B_BDBA = CritAnd(HasAtomNumber(6), HasNeighborNumbers(5, 6, 6))
C_B_BR_BDBA = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C_B_BDBA))
H_B_BR_BDBA = CritAnd(HasAtomNumber(1), HasNeighbors(C_B_BR_BDBA))

B1_term     =   CritAnd(HasAtomNumber(5), HasNeighbors(C_B_BDBA, HasAtomNumber(8), HasAtomNumber(8)))
O2_term     =   CritAnd(HasAtomNumber(8), HasNeighbors(HasAtomNumber(5), B1_term))
B3_term     =   CritAnd(HasAtomNumber(5), HasNeighborNumbers(6, 8, 8))
O4_term     =   CritAnd(HasAtomNumber(8), HasNeighbors(B3_term, B3_term))
C4_term     =   CritAnd(HasAtomNumber(6), HasNeighborNumbers(1, 1, 1, 5))
H5_term     =   CritAnd(HasAtomNumber(1), HasNeighbors(C4_term))

afilters = [('C_B_BDBA', C_B_BDBA), ('C_B_BR_BDBA', C_B_BR_BDBA), ('H_B_BR_BDBA', H_B_BR_BDBA), ('B1_term', B1_term), ('O2_term', O2_term), ('B3_term', B3_term), ('O4_term', O4_term), ('C4_term', C4_term), ('H5_term', H5_term)]

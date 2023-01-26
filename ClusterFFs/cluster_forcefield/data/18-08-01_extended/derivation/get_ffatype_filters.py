#! /usr/bin/env python

from molmod.molecular_graphs import MolecularGraph
from molmod.molecular_graphs import HasAtomNumber, HasNumNeighbors, HasNeighborNumbers, HasNeighbors
from molmod.graphs import CritAnd, CritNot, CritOr

B_HHTP = HasAtomNumber(5)
O_HHTP = HasAtomNumber(8)

C_O_HHTP = CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 8))
C_O_BR_HHTP = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C_O_HHTP))
C_BR_HHTP = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), HasAtomNumber(6), C_O_BR_HHTP))

H_O_BR_HHTP = CritAnd(HasAtomNumber(1), HasNeighbors(C_O_BR_HHTP))

C1_term = CritAnd(HasAtomNumber(6), HasNeighborNumbers(5, 6, 6))
C2_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C1_term))
C3_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), HasAtomNumber(6), C2_term))
C4_H_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C3_term))
C4_C_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), C3_term, C3_term))
C5_H_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C4_H_term))
C5_C_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), HasAtomNumber(6), C4_C_term))
C6_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), C5_H_term, C5_C_term))
C7_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C6_term))
C8_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), C7_term, C7_term))
H3_term = CritAnd(HasAtomNumber(1), HasNeighbors(C2_term))
H5_term = CritAnd(HasAtomNumber(1), HasNeighbors(C4_H_term))
H6_term = CritAnd(HasAtomNumber(1), HasNeighbors(C5_H_term))
H8_term = CritAnd(HasAtomNumber(1), HasNeighbors(C7_term))
H9_term = CritAnd(HasAtomNumber(1), HasNeighbors(C8_term))

afilters = [('B_HHTP', B_HHTP), ('O_HHTP', O_HHTP), ('C_O_HHTP', C_O_HHTP), ('C_O_BR_HHTP', C_O_BR_HHTP), ('C_BR_HHTP', C_BR_HHTP), ('H_O_BR_HHTP', H_O_BR_HHTP), ('C1_term', C1_term), ('C2_term', C2_term), ('C3_term', C3_term), ('C4_H_term', C4_H_term), ('C4_C_term', C4_C_term), ('C5_H_term', C5_H_term), ('C5_C_term', C5_C_term), ('C6_term', C6_term), ('C7_term', C7_term), ('C8_term', C8_term), ('H3_term', H3_term), ('H5_term', H5_term), ('H6_term', H6_term), ('H8_term', H8_term), ('H9_term', H9_term)]



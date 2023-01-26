#! /usr/bin/env python

from molmod.molecular_graphs import MolecularGraph
from molmod.molecular_graphs import HasAtomNumber, HasNumNeighbors, HasNeighborNumbers, HasNeighbors
from molmod.graphs import CritAnd, CritNot, CritOr

C_H     = CritAnd(HasAtomNumber(6), HasNeighborNumbers(1, 6, 6))
C4_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(5), CritAnd(C_H, HasNeighbors(C_H, HasAtomNumber(6), HasAtomNumber(1))), CritAnd(C_H, HasNeighbors(C_H, HasAtomNumber(6), HasAtomNumber(1)))))
C5_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), C4_term, C_H))
C6_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), C5_term, C_H))
C7_term = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), C6_term, C6_term))

B3_term  = CritAnd(HasAtomNumber(5), HasNeighbors(HasAtomNumber(8), HasAtomNumber(8), C4_term))
O4_term  = CritAnd(HasAtomNumber(8), HasNeighbors(B3_term, B3_term))
O2_term     = CritAnd(HasAtomNumber(8), HasNeighbors(CritAnd(HasAtomNumber(5), CritNot(B3_term)), B3_term))
B1_term     = CritAnd(HasAtomNumber(5), HasNeighbors(O2_term, O2_term, HasAtomNumber(6)))

C_B_PDBA     = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), HasAtomNumber(6), B1_term))
C_B_BR_O_PDBA= CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C_B_PDBA))
C_TP_PDBA    = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), HasAtomNumber(6), C_B_BR_O_PDBA))
C_TP_O_PDBA  = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(6), C_TP_PDBA))
C_TP_I_PDBA  = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), HasAtomNumber(6), C_TP_PDBA))

H6_term = CritAnd(HasAtomNumber(1), HasNeighbors(C5_term))
H7_term = CritAnd(HasAtomNumber(1), HasNeighbors(C6_term))
H8_term = CritAnd(HasAtomNumber(1), HasNeighbors(C7_term))
H_B_BR_O_PDBA= CritAnd(HasAtomNumber(1), HasNeighbors(C_B_BR_O_PDBA))
H_TP_O_PDBA  = CritAnd(HasAtomNumber(1), HasNeighbors(C_TP_O_PDBA))

afilters = [('C_B_PDBA', C_B_PDBA), ('C_B_BR_O_PDBA', C_B_BR_O_PDBA), ('C_TP_PDBA', C_TP_PDBA), ('C_TP_O_PDBA', C_TP_O_PDBA), ('C_TP_I_PDBA', C_TP_I_PDBA), ('H_B_BR_O_PDBA', H_B_BR_O_PDBA), ('H_TP_O_PDBA', H_TP_O_PDBA), ('B1_term', B1_term), ('O2_term', O2_term), ('B3_term', B3_term), ('O4_term', O4_term), ('C4_term', C4_term), ('C5_term', C5_term), ('C6_term', C6_term), ('C7_term', C7_term), ('H6_term', H6_term), ('H7_term', H7_term), ('H8_term', H8_term)]
#afilters = [('C0_term', C0_term), ('C1_term', C1_term), ('C2_term', C2_term), ('C3_term', C3_term), ('B_term', B_term), ('O_term', O_term), ('O_r', O_r), ('B_r', B_r), ('C_B', C_B), ('C_B_BR_O', C_B_BR_O), ('C_TP', C_TP), ('C_TP_O', C_TP_O), ('C_TP_I', C_TP_I), ('H1_term', H1_term), ('H2_term', H2_term), ('H3_term', H3_term), ('H_B_BR_O', H_B_BR_O), ('H_TP_O', H_TP_O)]






#C3_term = CritAnd(HasAtomNumber(6), HasNeighbors(1, C_H, C_H))
#C2_term = CritAnd(HasAtomNumber(6), HasNeighbors(1, 6, C3_term

#C3_term = CritAnd(HasAtomNumber(6), HasNeighbors(CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 1))))


#C3_term = CritAnd(HasAtomNumber(6), HasNeighbors(CritAnd(HasAtomNumber(6), HasNeighbors(CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(6), HasNeighbors()))))))
#C3_term = CritAnd(6, HasNeighbors(1, HasNeighbors(1, HasNeighbors(1, HasNeighbors()))))

#C_TP_I = CritAnd(HasAtomNumber(6), HasNeighbors(HasNeighborNumbers(6, 6, 6), HasNeighborNumbers(6, 6, 6), HasNeighborNumbers(6, 6, 6)))
#C_TP = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1), HasAtomNumber(1), C_TP_I))



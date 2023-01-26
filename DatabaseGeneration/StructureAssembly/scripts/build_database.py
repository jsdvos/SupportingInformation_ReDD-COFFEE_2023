import os

from molmod.units import angstrom, nanometer

from SBU import SBU
from Topology import Topology
from Database import Database

def get_sbus():
    sbus = []
    sbu_path = '../data/SBUs'
    for i in os.listdir(sbu_path):
        for j in os.listdir(sbu_path + i):
            sbu = SBU.load(i, j)
            sbu.load_parameters()
            sbus.append(sbu)
    return sbus

def get_topologies():
    topologies = []
    top_path = '../data/Topologies/'
    for dim in ['2D', '3D']:
        for fn_top in os.listdir(top_path + dim):
            top_name = fn_top.split('.')[0]
            top = Topology.load(top_name, dim)
            topologies.append(top)
    return topologies

if __name__ == '__main__':
    sbus = get_sbus()
    topologies = get_topologies()
    database = Database.from_sbus_topologies(sbus, topologies, max_possibilities = 10**4)
    database.compute(fn = '../data/database_resc_rmsd.txt')
    database.reduce(resc_tol = 0.22*angstrom, rmsd_tol = 0.11*angstrom)
    database.compute_properties(fn = '../data/database_natom_vinit.txt')
    database.reduce(natom_max = 10000, volume_max = 10000*nanometer**3)
    database.build('../data/Output')






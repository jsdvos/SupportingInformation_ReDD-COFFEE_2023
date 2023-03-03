from yaff import System, ForceField, angstrom, swap_noncovalent_lammps, kjmol, bar, VerletScreenLog, NHCThermostat, MTKBarostat, femtosecond, HDF5Writer, TBCombination, VerletIntegrator, kelvin, atm
import numpy as np
import h5py as h5
import time

def get_ff_kwargs():
    # Return force field arguments
    kwargs = {'rcut': 11.0*angstrom,
              'alpha_scale': 2.86,
              'gcut_scale': 1.0,
              'smooth_ei': True,
              'tailcorrections': True}
    return kwargs

mpi = False
def load_ff(sys, fn_pars, use_lammps = True):
    ff_kwargs = get_ff_kwargs()
    ff = ForceField.generate(sys, fn_pars, **ff_kwargs)
    if use_lammps:
        fn_sys = 'system_qff.dat' # LAMMPS System file
        fn_table = 'table_qff.dat' # LAMMPS force field tabulation file
        # Tabulate the non-bonded interactions
        # Bonded interactions remain calculated by Yaff
        if mpi:
            ff_lammps = swap_noncovalent_lammps(ff, fn_system = fn_sys,
                    fn_table = fn_table, comm = comm)
        else:
            ff_lammps = swap_noncovalent_lammps(ff, fn_system = fn_sys,
                    fn_table = fn_table)
        gpos, vtens = np.zeros((sys.natom, 3)), np.zeros((3, 3))
        gpos_lammps, vtens_lammps = np.zeros((sys.natom, 3)), np.zeros((3, 3))
        e = ff.compute(gpos, vtens)
        e_lammps = ff_lammps.compute(gpos_lammps, vtens_lammps)
        p = np.trace(vtens)/3.0/ff.system.cell.volume
        p_lammps = np.trace(vtens_lammps)/3.0/ff.system.cell.volume
        print("E(Yaff) = %12.3f E(LAMMPS) = %12.3f deltaE = %12.3e kJ/mol"%(e/kjmol,e_lammps/kjmol,(e_lammps-e)/kjmol))
        print("P(Yaff) = %12.3f P(LAMMPS) = %12.3f deltaP = %12.3e bar"%(p/bar,p_lammps/bar,(p_lammps-p)/bar))
        return ff_lammps
    else:
        return ff

for cof in ['COF-300', 'LZU-111', 'COF-320_89K', 'COF-320_298K']:
    if cof in ['COF-300', 'LZU-111']:
        temp = 100*kelvin
    elif cof == 'COF-320_89K':
        temp = 89*kelvin
    elif cof == 'COF-320_298K':
        temp = 298*kelvin
    press = 1*atm
    for ff_label in ['qff', 'uff']:
        if ff_label == 'qff':
            fn_pars = 'pars_cluster.txt'
        elif ff_label == 'uff':
            fn_pars = 'pars_uff.txt'
        else:
            raise NotImplementedError

        sys = System.from_file('../data/{}/{}_{}_opt.chk'.format(cof, cof, ff_label))
        ff = load_ff(sys, fn_pars)
        vsl = VerletScreenLog(step = 10)
        thermo = NHCThermostat(temp, timecon = 100.0*femtosecond)
        baro = MTKBarostat(ff, temp = temp, press = press, timecon = 1000*femtosecond)
        hdf = HDF5Writer(h5.File('../data/{}/{}_{}.h5'.format(cof, cof, ff_label), mode = 'w'), step = 100)
        tbc = TBCombination(thermo, baro)
        md = VerletIntegrator(ff, 0.5*femtosecond, temp0 = temp, hooks = [vsl, hdf, tbc])
        md.run(1000000)




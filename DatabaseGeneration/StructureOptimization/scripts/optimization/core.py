#!/usr/bin/env python

import os
import sys
from datetime import datetime

import numpy as np

from molmod.units import nanometer, angstrom, kjmol, bar

# Duplicates of yaff with log=None argument to allow for parallel runs
from yaff_system import System
from yaff_ff import ForceField
from yaff_liblammps import swap_noncovalent_lammps
from utils_log import Log
from yaff.sampling import CartesianDOF, StrainCellDOF # No log initialized, so OK?
from utils_application import CGOptimizer

__all__ = ['StructureOptimization']

def get_ff_kwargs():
    # Return force field arguments
    kwargs = {'rcut': 11.0*angstrom,
              'alpha_scale': 2.86,
              'gcut_scale': 1.0,
              'smooth_ei': True,
              'tailcorrections': True}
    return kwargs

def get_lammps_kwargs(natom, name, log, mpi = False):
    # Return lammps arguments
    kwargs = {'fn_table': '{}/table.dat'.format(name),'fn_system': '{}/system.dat'.format(name)}
    if natom > 2000:
        kwargs['kspace'] = 'pppm'
    if mpi:
        from mpi4py import MPI

        # Setup MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        # Turn off logging for all processes, except one
        log.set_level(log.silent)
        if rank == 0: log.set_level(log.medium)
        kwargs['comm'] = comm
    return kwargs

def get_dof_kwargs():
    return {}

def get_opt_kwargs():
    return {}

def get_n_save(natom):
    # After how many steps the structure is saved?
    if natom > 10**5:
        return 10
    elif natom > 10**3:
        return 100
    else:
        return 1000

def check_init_lammps(ff):
    for i in range(ff.nlist.nneigh):
        if ff.nlist.neighs[i]['d'] < 0.5*angstrom:
            a = ff.nlist.neighs[i]['a']
            b = ff.nlist.neighs[i]['b']
            d = ff.nlist.neighs[i]['d']
            log('Atoms {} and {} come to close ({}A)'.format(a, b, d/angstrom))
            log('This will cause an MPI Abort from LAMMPS')
            log('Continue with normal yaff force field')
            return False
    return True


def check_lammps(ff, ff_lammps, log):
    # Compare yaff and yaff+LAMMPS energies
    natom = ff.system.natom
    assert ff_lammps.system.natom == natom
    gpos = np.zeros((natom, 3))
    gpos_lammps = np.zeros((natom, 3))
    vtens = np.zeros((3, 3))
    vtens_lammps = np.zeros((3, 3))
    e = ff.compute(gpos, vtens)
    e_lammps = ff_lammps.compute(gpos_lammps, vtens_lammps)
    p = np.trace(vtens)/3.0/ff.system.cell.volume
    p_lammps = np.trace(vtens_lammps)/3.0/ff_lammps.system.cell.volume
    log('Generated LAMMPS force part:')
    log('\tE(Yaff) = {:.3f}kJ/mol'.format(e/kjmol))
    log('\tE(LAMMPS) = {:.3f}kJ/mol'.format(e_lammps/kjmol))
    log('\tP(Yaff) = {:.3f}bar'.format(p/bar))
    log('\tP(LAMMPS) = {:.3f}bar'.format(p_lammps/bar))
    if abs(e-e_lammps) > 5*kjmol or abs(p-p_lammps) > 5*bar:
        message = 'Yaff and LAMMPS do not return the same energies:\n'
        message += '\t\tE(Yaff) = {: .3f}kJ/mol\t\tE(LAMMPS) = {: .3f}kJ/mol\n'.format(e/kjmol, e_lammps/kjmol)
        message += '\t\tP(Yaff) = {: .3f}bar\t\tP(LAMMPS) = {: .3f}bar'.format(p/bar, p_lammps/bar)
        return False, message
    else:
        return True, ""

class StructureOptimization(object):
    def __init__(self, name, only_yaff = False):
        # I/O files
        self.name = name
        self.only_yaff = only_yaff
        
        self.fn_system = '{}/{}.chk'.format(self.name, self.name)
        self.fn_uff_opt = '{}/{}_init.chk'.format(self.name, self.name)
        self.fn_cartquickff_opt = '{}/system_optimized.chk'.format(self.name)
        self.fn_opt = '{}/{}_optimized.chk'.format(self.name, self.name)
        self.fn_uff_save = '{}/{}_init_save.chk'.format(self.name, self.name)
        self.fn_save = '{}/{}_save.chk'.format(self.name, self.name)
        self.fn_uff = '{}/pars_uff.txt'.format(self.name)
        self.fn_quickff = '{}/pars_cluster.txt'.format(self.name)
        self.fn_log_status = '{}/{}_OPT.log'.format(self.name, self.name)
        self.fn_log = '{}/opt.log'.format(self.name)

        # Status
        self.status = self.get_status()

        # Initialize a custom log and timer object to allow multiprocessing with dask
        log, timer = Log()
        self.log = log
        self.timer = timer

    def get_status(self):
        '''
        Find the status of the optimization. Following integers are defined:
        -1: status not yet defined
        0: No optimization has finalized
        1: CartesianDOF UFF optimalisation (50 steps) is done
        2: CartesianDOF QuickFF optimalisation is done
        3: StrainCellDOF QuickFF optimalisation is done - fully optimized
        '''
        if os.path.exists(self.fn_opt):
            return 3
        if not os.path.exists(self.fn_log_status):
            return 0
        with open(self.fn_log_status, 'r') as f:
            log_status = f.read()
        if os.path.exists(self.fn_cartquickff_opt):
            return 2
        if 'StrainCellDOF None Starting' in log_status and os.path.exists(self.fn_save):
            return 2
        if os.path.exists(self.fn_uff_opt):
            return 1
        if 'CartesianDOF None Starting' in log_status and os.path.exists(self.fn_save):
            return 1
        if os.path.exists(self.fn_save):
            return 1
        return 0

    def load_ff(self, system, parameters, noncovalent_lammps = False):
        '''
        Arguments:
            system: System object
            parameters: Parameters object or files from which it is made
            ff_kwargs can contain
                rcut: real space cutoff
                tr (truncation model for every pair potential except the ei)
                alpha_scale
                gcut_scale
                smooth_ei
                tailcorrections
                reci_ei (Method to compute reciprocal contributions)
                skin
                nlow
                nhigh
            lammps_kwargs can contain
                fn_system
                fn_table
                fn_log
                suffix (suffix of liblammps_*.so library)
                do_table
                overwrite_table (update if already exists?)
                nrows (number of rows for tabulating noncovalent interactions)
                keep_forceparts (forceparts computed by yaff)
                scalings_vdw
                scalings_ei
                do_ei
                kspace ('ewald' or 'pppm')
                kspace_accuracy
                triclinic
                comm
                move_central_cell (change to True if Atoms Lost ERROR)
        '''
        ff_kwargs = get_ff_kwargs()
        ff = ForceField.generate(system, parameters, log = self.log, timer = self.timer, **ff_kwargs)
        if noncovalent_lammps and check_init_lammps(ff):
            lammps_kwargs = get_lammps_kwargs(system.natom, self.name, log = self.log)
            try:
                try:
                    ff_lammps = swap_noncovalent_lammps(ff, **lammps_kwargs)
                    flag, msg = check_lammps(ff, ff_lammps, log = self.log)
                except Exception as e:
                    # System is to small to create LAMMPS table, try using a supercell
                    self.log('Encountered following error:\n{}'.format(e))
                    self.log('Possibly this error is due to a small system, trying to use supercell..')
                    super_system = system.supercell(2,2,2)
                    super_ff = ForceField.generate(super_system, parameters, log = self.log, timer = self.timer, **ff_kwargs)
                    lammps_kwargs['overwrite_table'] = True
                    super_ff_lammps = swap_noncovalent_lammps(super_ff, **lammps_kwargs)
                    # Generate the force field for the unit cell with the table generated on the supercell
                    lammps_kwargs['overwrite_table'] = False
                    ff_lammps = swap_noncovalent_lammps(ff, **lammps_kwargs)
                    flag, msg = check_lammps(ff, ff_lammps, log = self.log)
                if not flag:
                    raise RuntimeError(msg)
                ff = ff_lammps
            except Exception as e:
                self.log('LAMMPS force field is not reliable or could not be generated due to following error:\n{}'.format(e))
                self.log('Continue with normal yaff force field')
                self.log.blank()
        return ff

    def optimize(self, ff):
        dof_kwargs = get_dof_kwargs()
        opt_kwargs = get_opt_kwargs()
        n_save = get_n_save(ff.system.natom)
        if self.status == 0:
            dof_class = CartesianDOF
            fn_save = self.fn_uff_save
            fn_out = self.fn_uff_opt
            nsteps = 50
        elif self.status == 1:
            dof_class = CartesianDOF
            fn_save = self.fn_save
            fn_out = self.fn_cartquickff_opt
            nsteps = None
        elif self.status == 2:
            dof_class = StrainCellDOF
            fn_save = self.fn_save
            fn_out = self.fn_opt
            nsteps = None
        elif self.status == 3:
            dof_class = StrainCellDOF
            fn_save = self.fn_save
            fn_out = self.fn_opt
            nsteps = None
        else:
            self.log('Does not recognize status {}'.format(self.status))
        dof = dof_class(ff, **dof_kwargs)
        opt = CGOptimizer(dof, **opt_kwargs)
        if self.fn_log_status is not None:
            with open(self.fn_log_status, 'a') as f:
                f.write('############   Starting   ############\n')
                f.write('{} {} {} {}\n'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                                               dof.__class__.__name__, nsteps, 'Starting'))

        # Run the optimizer, saving intermediate structure if asked
        while not dof.converged and (nsteps == None or nsteps > 0):
            if nsteps is None and n_save is None:
                # Optimize until converged
                n_run = None
            elif nsteps is None and n_save is not None:
                # Optimize until converged, but keep track of structure
                n_run = n_save
            elif nsteps is not None and n_save is None:
                # Optimze N steps
                n_run = nsteps
                nsteps -= n_run
            elif nsteps is not None and n_save is not None:
                # Optimize N steps and keep track of structure
                if nsteps > n_save:
                    n_run = n_save
                else:
                    n_run = nsteps
                nsteps -= n_run
            # Store some values to compare later
            counter = opt.counter
            conv_val = opt.dof.conv_val
            # Do the magic
            opt.run(n_run)
            if n_save is not None:
                opt.dof.ff.system.to_file(fn_save)
            if self.fn_log_status is not None:
                with open(self.fn_log_status, 'a') as f:
                    f.write('{} {} {} {}\n'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                                               dof.__class__.__name__, opt.counter, dof.converged))

            if not dof.converged:
                # Check if not crashed
                if not counter + n_run == opt.counter:
                    # Check 1: something interrupted optimization run
                    if self.fn_log_status is not None:
                        with open(self.fn_log_status, 'a') as f:
                            f.write('{} {} {} {}\n'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                                                   dof.__class__.__name__, opt.counter, 'Interrupted'))
                    raise RuntimeError('Interrupted optimization run', opt.dof.conv_val)
                if np.around(conv_val, 6) == np.around(opt.dof.conv_val, 6):
                    # Check 2: optimizer is not improving anymore
                    # To avoid that the convergence value is not the same by coincidence,
                    # we run 10 steps more and look if the convergence value changes
                    for i in range(10):
                        opt.run(1)
                        if not np.around(conv_val, 6) == np.around(opt.dof.conv_val, 6):
                            break
                    else:
                        if self.fn_log_status is not None:
                            with open(self.fn_log_status, 'a') as f:
                                f.write('{} {} {} {}\n'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                                                       dof.__class__.__name__, opt.counter, 'NonImproving'))
                        raise RuntimeError('NonImproving optimization run', opt.dof.conv_val)
        
        if dof.converged or (self.status == 0 and opt.counter == 50):
            self.status += 1
            if fn_out is not None:
                opt.dof.ff.system.to_file(fn_out)
            if dof.converged and self.fn_log_status is not None:
                with open(self.fn_log_status, 'a') as f:
                    f.write('{} {} {} {}\n'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                                                dof.__class__.__name__, opt.counter, 'Converged'))
        if self.fn_log_status is not None:
            with open(self.fn_log_status, 'a') as f:
                f.write('############  Terminated  ############\n\n')


    def run(self):
        with open(self.fn_log, 'a') as fn_out:
            self.log.set_file(fn_out)
            
            if self.status < 1:
                fn_system = self.fn_system
                fn_save = self.fn_uff_save
            else:
                fn_system = self.fn_uff_opt
                fn_save = self.fn_save
            
            if os.path.exists(fn_save):
                self.log('Saved structure detected, switching to this system')
                system = System.from_file(fn_save, log = self.log)
            else:
                system = System.from_file(fn_system, log = self.log)
            if self.only_yaff:
                self.log('LAMMPS force field is turned OFF')

            ff = None
            while self.status < 3:
                print('Starting {} with status {}'.format(self.name, self.status))
                old_status = self.status
                if self.status < 1:
                    fn_pars = self.fn_uff
                    noncovalent_lammps = False
                else:
                    fn_pars = self.fn_quickff
                    if self.only_yaff:
                        noncovalent_lammps = False
                    else:
                        noncovalent_lammps = True
                if self.status < 2 or ff is None:
                    ff = self.load_ff(system, fn_pars, noncovalent_lammps = noncovalent_lammps)
                try:
                    self.optimize(ff)
                except RuntimeError as e:
                    self.log('Encountered following RuntimeError during optimization:')
                    self.log('{} (conv_val={})'.format(e.args[0], e.args[1]))
                    if 'lammps' in [part.name for part in ff.parts]:
                        self.log('Switch to pure yaff force field')
                        ff = self.load_ff(system, fn_pars, noncovalent_lammps = False)
                        self.optimize(ff)
                    else:
                        raise e

                if not self.status == old_status + 1:
                    sys.stdout.flush()
                    raise RuntimeError('Failed at status {}, final status stays {}'.format(old_status, self.status))
                print('{} continues with status {}'.format(self.name, self.status))
                sys.stdout.flush()
            self.log.print_footer()

"""if __name__ == '__main__':
    # Setup run
    args = sys.argv
    name = args[1]
    only_yaff = False
    if len(args) > 2:
        if args[2] == 'only_yaff':
            only_yaff = True
            print('LAMMPS force field turned OFF')
        else:
            print('Dont know what to do with arguments (got: {})'.format(args))
            print('Continue with {} as fn_input and without only_yaff'.format(fn_input))

    # Run
    optimizer = StructureOptimization(name, only_yaff = only_yaff)
    optimizer.run()
"""

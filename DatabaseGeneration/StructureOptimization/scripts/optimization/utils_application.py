#!/usr/bin/env python
from __future__ import division
import numpy as np

from molmod.units import *
from molmod.constants import *

from molmod.minimizer import ConjugateGradient, NewtonLineSearch, Minimizer

from yaff.sampling.nvt import NHCThermostat
from yaff.sampling.npt import MTKBarostat, TBCombination
from yaff.sampling.iterative import Hook, StateItem, AttributeStateItem, PosStateItem,\
                                    DipoleStateItem, DipoleVelStateItem, VolumeStateItem, \
                                    CellStateItem, EPotContribStateItem
from yaff.sampling.verlet import TemperatureStateItem, VerletHook, ConsErrTracker
from yaff.sampling.utils import get_random_vel

from utils_log import OptScreenLog

#########################
# Iterative and VerletIntegrator with log and timer attribute

class Iterative(object):
    default_state = []
    log_name = 'ITER'

    def __init__(self, ff, state=None, hooks=None, counter0=0):
        """
           **Arguments:**

           ff
                The ForceField instance used in the iterative algorithm

           **Optional arguments:**

           state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

           hooks
                A function (or a list of functions) that is called after every
                iterative.

           counter0
                The counter value associated with the initial state.
        """
        self.ff = ff
        if state is None:
            self.state_list = [state_item.copy() for state_item in self.default_state]
        else:
            #self.state_list = state
            self.state_list = [state_item.copy() for state_item in self.default_state]
            self.state_list += state
        self.state = dict((item.key, item) for item in self.state_list)
        if hooks is None:
            self.hooks = []
        elif hasattr(hooks, '__len__'):
            self.hooks = hooks
        else:
            self.hooks = [hooks]
        self._add_default_hooks()
        self.counter0 = counter0
        self.counter = counter0
        with self.ff.log.section(self.log_name), self.ff.timer.section(self.log_name):
            self.initialize()

        # Initialize restart hook if present
        from yaff.sampling.io import RestartWriter
        for hook in self.hooks:
            if isinstance(hook, RestartWriter):
                hook.init_state(self)

    def _add_default_hooks(self):
        pass

    def initialize(self):
        self.call_hooks()

    def call_hooks(self):
        with self.ff.timer.section('%s hooks' % self.log_name):
            state_updated = False
            from yaff.sampling.io import RestartWriter
            for hook in self.hooks:
                if hook.expects_call(self.counter) and not (isinstance(hook, RestartWriter) and self.counter==self.counter0):
                    if not state_updated:
                        for item in self.state_list:
                            item.update(self)
                        state_updated = True
                    if isinstance(hook, RestartWriter):
                        for item in hook.state_list:
                            item.update(self)
                    hook(self)

    def run(self, nstep=None):
        with self.ff.log.section(self.log_name), self.ff.timer.section(self.log_name):
            if nstep is None:
                while True:
                    if self.propagate():
                        break
            else:
                for i in range(nstep):
                    if self.propagate():
                        break
            self.finalize()

    def propagate(self):
        self.counter += 1
        self.call_hooks()

    def finalize():
        raise NotImplementedError



class VerletIntegrator(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('time'),
        AttributeStateItem('epot'),
        PosStateItem(),
        AttributeStateItem('vel'),
        AttributeStateItem('rmsd_delta'),
        AttributeStateItem('rmsd_gpos'),
        AttributeStateItem('ekin'),
        TemperatureStateItem(),
        AttributeStateItem('etot'),
        AttributeStateItem('econs'),
        AttributeStateItem('cons_err'),
        AttributeStateItem('ptens'),
        AttributeStateItem('vtens'),
        AttributeStateItem('press'),
        DipoleStateItem(),
        DipoleVelStateItem(),
        VolumeStateItem(),
        CellStateItem(),
        EPotContribStateItem(),
    ]

    log_name = 'VERLET'

    def __init__(self, ff, timestep=None, state=None, hooks=None, vel0=None,
                 temp0=300, scalevel0=True, time0=None, ndof=None, counter0=None, restart_h5=None):
        """
            **Arguments:**

            ff
                A ForceField instance

            **Optional arguments:**

            timestep
                The integration time step (in atomic units)

            state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

            hooks
                A function (or a list of functions) that is called after every
                iterative.

            vel0
                An array with initial velocities. If not given, random
                velocities are sampled from the Maxwell-Boltzmann distribution
                corresponding to the optional arguments temp0 and scalevel0

            temp0
                The (initial) temperature for the random initial velocities

            scalevel0
                If True (the default), the random velocities are rescaled such
                that the instantaneous temperature coincides with temp0.

            time0
                The time associated with the initial state.

            ndof
                When given, this option overrides the number of degrees of
                freedom determined from internal heuristics. When ndof is not
                given, its default value depends on the thermostat used. In most
                cases it is 3*natom, except for the NHC thermostat where the
                number if internal degrees of freedom is counted. The ndof
                attribute is used to derive the temperature from the kinetic
                energy.

            counter0
                The counter value associated with the initial state.

            restart_h5
                HDF5 object containing the restart information
        """
        # Assign init arguments
        if timestep is None and restart_h5 is None:
            raise AssertionError('No Verlet timestep is found')
        self.ndof = ndof
        self.hooks = hooks
        self.restart_h5 = restart_h5
        self.log = ff.log
        self.timer = ff.timer

        # Retrieve the necessary properties if restarting. Restart objects
        # are overwritten by optional arguments in VerletIntegrator
        if self.restart_h5 is None:
            # set None variables to default value
            if time0 is None: time0 = 0.0
            if counter0 is None: counter0 = 0
            self.pos = ff.system.pos.copy()
            self.rvecs = ff.system.cell.rvecs.copy()
            self.timestep = timestep
            self.time = time0
        else:
            # Arguments associated with the unit cell and positions are always retrieved
            tgrp = self.restart_h5['trajectory']
            self.pos = tgrp['pos'][-1,:,:]
            ff.update_pos(self.pos)
            if 'cell' in tgrp:
                self.rvecs = tgrp['cell'][-1,:,:]
                ff.update_rvecs(self.rvecs)
            else:
                self.rvecs = None
            # Arguments which can be provided in the VerletIntegrator object are only
            # taken from the restart file if not provided explicitly
            if time0 is None:
                self.time = tgrp['time'][-1]
            else:
                self.time = time0
            if counter0 is None:
                counter0 = tgrp['counter'][-1]
            if vel0 is None:
                vel0 = tgrp['vel'][-1,:,:]
            if timestep is None:
                self.timestep = self.restart_h5['/restart/timestep'][()]
            self._restart_add_hooks(self.restart_h5, ff)

        # Verify the hooks: combine thermostat and barostat if present
        self._verify_hooks()

        # The integrator needs masses. If not available, take default values.
        if ff.system.masses is None:
            ff.system.set_standard_masses()
        self.masses = ff.system.masses

        # Set random initial velocities if needed.
        if vel0 is None:
            self.vel = get_random_vel(temp0, scalevel0, self.masses)
        else:
            self.vel = vel0.copy()

        # Working arrays
        self.gpos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)
        self.vtens = np.zeros((3, 3), float)

        # Tracks quality of the conserved quantity
        self._cons_err_tracker = ConsErrTracker(restart_h5)
        Iterative.__init__(self, ff, state, self.hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, VerletScreenLog) for hook in self.hooks):
            self.hooks.append(VerletScreenLog(log=self.log))

    def _verify_hooks(self):
        with self.log.section('ENSEM'):
            thermo = None
            index_thermo = 0
            baro = None
            index_baro = 0

            # Look for the presence of a thermostat and/or barostat
            if hasattr(self.hooks, '__len__'):
                for index, hook in enumerate(self.hooks):
                    if hook.method == 'thermostat':
                        thermo = hook
                        index_thermo = index
                    elif hook.method == 'barostat':
                        baro = hook
                        index_baro = index
            elif self.hooks is not None:
                if self.hooks.method == 'thermostat':
                    thermo = self.hooks
                elif self.hooks.method == 'barostat':
                    baro = self.hooks

            # If both are present, delete them and generate TBCombination element
            if thermo is not None and baro is not None:
                from yaff.sampling.npt import TBCombination
                if self.log.do_warning:
                    self.log.warn('Both thermostat and barostat are present separately and will be merged')
                del self.hooks[max(index_thermo, index_thermo)]
                del self.hooks[min(index_thermo, index_baro)]
                self.hooks.append(TBCombination(thermo, baro))

            if hasattr(self.hooks, '__len__'):
                for hook in self.hooks:
                    if hook.name == 'TBCombination':
                        thermo = hook.thermostat
                        baro = hook.barostat
            elif self.hooks is not None:
                if self.hooks.name == 'TBCombination':
                    thermo = self.hooks.thermostat
                    baro = self.hooks.barostat

            if self.log.do_warning:
                if thermo is not None:
                    self.log('Temperature coupling achieved through ' + str(thermo.name) + ' thermostat')
                if baro is not None:
                    self.log('Pressure coupling achieved through ' + str(baro.name) + ' barostat')

    def _restart_add_hooks(self, restart_h5, ff):
        # First, make sure that no thermostat / barostat hooks are supplied in the hooks argument.
        # If this is the case, they are NOT overwritten.
        thermo = None
        baro = None
        baro_thermo = None
        for hook in self.hooks:
            if hook.method == 'thermostat': thermo = hook
            elif hook.method == 'barostat': baro = hook
            elif hook.name == 'TBCombination':
                thermo = hook.thermostat
                baro = hook.barostat

        if thermo is None or baro is None: # not all hooks are already provided
            rgrp = restart_h5['/restart']
            tgrp = restart_h5['/trajectory']

            # verify if NHC thermostat is present
            if thermo is None and 'thermo_name' in rgrp:
                # collect thermostat properties and create thermostat
                thermo_name = rgrp['thermo_name'][()]
                timecon = rgrp['thermo_timecon'][()]
                temp = rgrp['thermo_temp'][()]
                if thermo_name == 'NHC':
                    pos0 = tgrp['thermo_pos'][-1,:]
                    vel0 = tgrp['thermo_vel'][-1,:]
                    thermo = NHCThermostat(temp, timecon=timecon, chainlength=len(pos0), chain_pos0=pos0, chain_vel0=vel0, restart=True)

            # verify if barostat is present
            if baro is None and 'baro_name' in rgrp:
                baro_name = rgrp['baro_name'][()]
                # if MTTK barostat, verify if barostat thermostat is present
                if baro_name == 'MTTK' and 'baro_chain_timecon' in rgrp:
                    # collect barostat thermostat properties
                    timecon = rgrp['baro_chain_timecon'][()]
                    temp = rgrp['baro_chain_temp'][()]
                    pos0 = tgrp['baro_chain_pos'][-1,:]
                    vel0 = tgrp['baro_chain_vel'][-1,:]
                    # create thermostat instance
                    baro_thermo = NHCThermostat(temp, timecon=timecon, chainlength=len(pos0), chain_pos0=pos0, chain_vel0=vel0, restart=True)
                # collect barostat properties and create barostat
                timecon = rgrp['baro_timecon'][()]
                temp = rgrp['baro_temp'][()]
                press = rgrp['baro_press'][()]
                anisotropic = rgrp['baro_anisotropic'][()]
                vol_constraint = rgrp['vol_constraint'][()]
                if baro_name == 'MTTK':
                    if anisotropic:
                        vel0 = tgrp['baro_vel_press'][-1,:,:]
                    else:
                        vel0 = tgrp['baro_vel_press'][-1]
                    baro = MTKBarostat(ff, temp, press, timecon=timecon, anisotropic=anisotropic, vol_constraint=vol_constraint, baro_thermo=baro_thermo, vel_press0=vel0, restart=True)

        # append the necessary hooks
        if thermo is not None and baro is not None:
            self.hooks.append(TBCombination(thermo, baro))
        elif thermo is not None:
            self.hooks.append(thermo)
        elif baro is not None:
            self.hooks.append(baro)

    def initialize(self):
        # Standard initialization of Verlet algorithm
        self.gpos[:] = 0.0
        self.ff.update_pos(self.pos)
        self.epot = self.ff.compute(self.gpos)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.posoud = self.pos.copy()

        # Allow for specialized initializations by the Verlet hooks.
        self.call_verlet_hooks('init')

        # Configure the number of degrees of freedom if needed
        if self.ndof is None:
            self.ndof = self.pos.size

        # Common post-processing of the initialization
        self.compute_properties(self.restart_h5)
        Iterative.initialize(self) # Includes calls to conventional hooks

    def propagate(self):
        # Allow specialized hooks to modify the state before the regular verlet
        # step.
        self.call_verlet_hooks('pre')

        # Regular verlet step
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*self.acc*self.timestep
        self.pos += self.timestep*self.vel
        self.ff.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.vtens[:] = 0.0
        self.epot = self.ff.compute(self.gpos, self.vtens)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*self.acc*self.timestep
        self.ekin = self._compute_ekin()

        # Allow specialized verlet hooks to modify the state after the step
        self.call_verlet_hooks('post')

        # Calculate the total position change
        self.posnieuw = self.pos.copy()
        self.delta[:] = self.posnieuw-self.posoud
        self.posoud[:] = self.posnieuw

        # Common post-processing of a single step
        self.time += self.timestep
        self.compute_properties()
        Iterative.propagate(self) # Includes call to conventional hooks

    def _compute_ekin(self):
        '''Auxiliary routine to compute the kinetic energy

           This is used internally and often also by the Verlet hooks.
        '''
        return 0.5*(self.vel**2*self.masses.reshape(-1,1)).sum()

    def compute_properties(self, restart_h5=None):
        self.rmsd_gpos = np.sqrt((self.gpos**2).mean())
        self.rmsd_delta = np.sqrt((self.delta**2).mean())
        self.ekin = self._compute_ekin()
        self.temp = self.ekin/self.ndof*2.0/boltzmann
        self.etot = self.ekin + self.epot
        self.econs = self.etot
        for hook in self.hooks:
            if isinstance(hook, VerletHook):
                self.econs += hook.econs_correction
        if restart_h5 is not None:
            self.econs = restart_h5['trajectory/econs'][-1]
        else:
            self._cons_err_tracker.update(self.ekin, self.econs)
        self.cons_err = self._cons_err_tracker.get()
        if self.ff.system.cell.nvec > 0:
            self.ptens = (np.dot(self.vel.T*self.masses, self.vel) - self.vtens)/self.ff.system.cell.volume
            self.press = np.trace(self.ptens)/3

    def finalize(self):
        if self.log.do_medium:
            self.log.hline()

    def call_verlet_hooks(self, kind):
        # In this call, the state items are not updated. The pre and post calls
        # of the verlet hooks can rely on the specific implementation of the
        # VerletIntegrator and need not to rely on the generic state item
        # interface.
        with self.timer.section('%s special hooks' % self.log_name):
            for hook in self.hooks:
                if isinstance(hook, VerletHook) and hook.expects_call(self.counter):
                    if kind == 'init':
                        hook.init(self)
                    elif kind == 'pre':
                        hook.pre(self)
                    elif kind == 'post':
                        hook.post(self)
                    else:
                        raise NotImplementedError

    def call_hooks(self):
        with self.timer.section('%s hooks' % self.log_name):
            state_updated = False
            from yaff.sampling.io import RestartWriter
            for hook in self.hooks:
                if hook.expects_call(self.counter) and not (isinstance(hook, RestartWriter) and self.counter==self.counter0):
                    if not state_updated:
                        for item in self.state_list:
                            item.update(self)
                        state_updated = True
                    if isinstance(hook, RestartWriter):
                        for item in hook.state_list:
                            item.update(self)
                    hook(self)



#########################
# Opt modules with log and timer attribute

class BaseOptimizer(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('epot'),
        PosStateItem(),
        DipoleStateItem(),
        VolumeStateItem(),
        CellStateItem(),
        EPotContribStateItem(),
    ]
    log_name = 'XXOPT'

    def __init__(self, dof, state=None, hooks=None, counter0=0):
        """
           **Arguments:**

           dof
                A specification of the degrees of freedom. The convergence
                criteria are also part of this argument. This must be a DOF
                instance.

           **Optional arguments:**

           state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

           hooks
                A function (or a list of functions) that is called after every
                iterative.

           counter0
                The counter value associated with the initial state.
        """
        self.dof = dof

        # Monkey patch the BaseCellDof
        def log_with_object(self):
            rvecs = self.ff.system.cell.rvecs
            lengths, angles = self.ff.system.cell.parameters
            rvec_names = 'abc'
            angle_names = ['alpha', 'beta', 'gamma']
            self.ff.log(" ")
            self.ff.log("Final Unit Cell:")
            self.ff.log("----------------")
            self.ff.log("- cell vectors:")
            for i in range(len(rvecs)):
                self.ff.log("    %s = %s %s %s" %(rvec_names[i], self.ff.log.length(rvecs[i,0]), self.ff.log.length(rvecs[i,1]), self.ff.log.length(rvecs[i,2]) ))
            self.ff.log(" ")
            self.ff.log("- lengths, angles and volume:")
            for i in range(len(rvecs)):
                self.ff.log("    |%s|  = %s" % (rvec_names[i], self.ff.log.length(lengths[i])))
            for i in range(len(angles)):
                self.ff.log("    %5s = %s" % (angle_names[i], self.ff.log.angle(angles[i])))
            self.ff.log("    volume = %s" % self.ff.log.volume(self.ff.system.cell.volume) )


        from yaff.sampling.dof import BaseCellDOF
        BaseCellDOF.log = log_with_object

        Iterative.__init__(self, dof.ff, state, hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, OptScreenLog) for hook in self.hooks):
            self.hooks.append(OptScreenLog(log=self.dof.ff.log))

    def fun(self, x, do_gradient=False):
        if do_gradient:
            self.epot, gx = self.dof.fun(x, True)
            return self.epot, gx
        else:
            self.epot = self.dof.fun(x, False)
            return self.epot

    def initialize(self):
        # The first call to check_convergence will never flag convergence, but
        # it is need to keep track of some convergence criteria.
        self.dof.check_convergence()
        Iterative.initialize(self)

    def propagate(self):
        self.dof.check_convergence()
        Iterative.propagate(self)
        return self.dof.converged

    def finalize(self):
        if self.dof.ff.log.do_medium:
            self.dof.log()
            self.dof.ff.log.hline()


class CGOptimizer(BaseOptimizer):
    log_name = 'CGOPT'

    def __init__(self, dof, state=None, hooks=None, counter0=0):
        self.minimizer = Minimizer(
            dof.x0, self.fun, ConjugateGradient(), NewtonLineSearch(), None,
            None, anagrad=True, verbose=False,
        )
        #log.set_level(log.medium)
        BaseOptimizer.__init__(self, dof, state, hooks, counter0)

    def initialize(self):
        self.minimizer.initialize()
        BaseOptimizer.initialize(self)

    def propagate(self):
        success = self.minimizer.propagate()
        self.x = self.minimizer.x
        if success == False:
            if self.dof.ff.log.do_warning:
                self.dof.ff.log.warn('Line search failed in optimizer. Aborting optimization. This is probably due to a dicontinuity in the energy or the forces. Check the truncation of the non-bonding interactions and the Ewald summation parameters.')
            return True
        return BaseOptimizer.propagate(self)

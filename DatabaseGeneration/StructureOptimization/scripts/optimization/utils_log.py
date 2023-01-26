#!/usr/bin/env python
from __future__ import division
import atexit

import numpy as np
import time

from molmod.log import SubTimer, ScreenLog
from contextlib import contextmanager

from yaff.sampling.iterative import Hook
from yaff import __version__ as yaff_version
from yaff.log import head_banner, foot_banner

#########################
# Custom Log function to return a log and timer object for Yaff/molmod functions

def Log():
    timer = TimerGroup()
    log = ScreenLog('YAFF', yaff_version, head_banner, foot_banner, timer)
    #log.set_level(0)
    return log, timer


#########################
# Custom VerletScreenLog class with log attribute

class VerletScreenLog(Hook):
    '''A screen logger for the Verlet algorithm'''
    def __init__(self, start=0, step=1, log=None):
        Hook.__init__(self, start, step)
        self.time0 = None
        self.log=log

    def __call__(self, iterative):
        if self.log.do_medium:
            if self.time0 is None:
                self.time0 = time.time()
                if self.log.do_medium:
                    self.log.hline()
                    self.log('Cons.Err. =&the root of the ratio of the variance on the conserved quantity and the variance on the kinetic energy.')
                    self.log('d-rmsd    =&the root-mean-square displacement of the atoms.')
                    self.log('g-rmsd    =&the root-mean-square gradient of the energy.')
                    self.log('counter  Cons.Err.       Temp     d-RMSD     g-RMSD   Walltime')
                    self.log.hline()
            self.log('%7i %10.5f %s %s %s %10.1f' % (
                iterative.counter,
                iterative.cons_err,
                self.log.temperature(iterative.temp),
                self.log.length(iterative.rmsd_delta),
                self.log.force(iterative.rmsd_gpos),
                time.time() - self.time0,
            ))


class OptScreenLog(Hook):
    def __init__(self, start=0, step=1, log=None):
        Hook.__init__(self, start, step)
        self.time0 = None
        self.log = log

    def __call__(self, iterative):
        if self.log.do_medium:
            if self.time0 is None:
                self.time0 = time.time()
                if self.log.do_medium:
                    self.log.hline()
                    self.log('Conv.val. =&the highest ratio of a convergence criterion over its threshold.')
                    self.log('N         =&the number of convergence criteria that is not met.')
                    self.log('Worst     =&the name of the convergence criterion that is worst.')
                    self.log('counter  Conv.val.  N           Worst     Energy   Walltime')
                    self.log.hline()
            self.log('%7i % 10.3e %2i %15s %s %10.1f' % (
                iterative.counter,
                iterative.dof.conv_val,
                iterative.dof.conv_count,
                iterative.dof.conv_worst,
                self.log.energy(iterative.epot),
                time.time() - self.time0,
            ))


#########################
# Custom TimerGroup class to allow some flexibility in the error handling

class TimerGroup(object):
    def __init__(self):
        self.parts = {}
        self._stack = []
        self._start('Total')

    def reset(self):
        for timer in self.parts.values():
            timer.total.cpu = 0.0
            timer.own.cpu = 0.0

    @contextmanager
    def section(self, label):
        self._start(label)
        try:
            yield
        finally:
            self._stop(label)

    def _start(self, label):
        # get the right timer object
        timer = self.parts.get(label)
        if timer is None:
            timer = SubTimer(label)
            self.parts[label] = timer
        # start timing
        timer.start()
        if len(self._stack) > 0:
            self._stack[-1].start_sub()
        # put it on the stack
        self._stack.append(timer)

    def _stop(self, label):
        timer = self._stack.pop(-1)
        try:
            assert timer.label == label
        except AssertionError:
            print(timer.label, '-', label)
        timer.stop()
        if len(self._stack) > 0:
            self._stack[-1].stop_sub()

    def get_max_own_cpu(self):
        result = None
        for part in self.parts.values():
            if result is None or result < part.own.cpu:
                result = part.own.cpu
        return result

    def report(self, log):
        max_own_cpu = self.get_max_own_cpu()
        #if max_own_cpu == 0.0:
        #    return
        with log.section('TIMER'):
            log('Overview of CPU time usage.')
            log.hline()
            log('Label             Total      Own')
            log.hline()
            bar_width = log.width-33
            for label, timer in sorted(self.parts.items()):
                #if timer.total.cpu == 0.0:
                #    continue
                if max_own_cpu > 0:
                    cpu_bar = "W"*int(timer.own.cpu/max_own_cpu*bar_width)
                else:
                    cpu_bar = ""
                log('%14s %8.1f %8.1f %s' % (
                    label.ljust(14),
                    timer.total.cpu, timer.own.cpu, cpu_bar.ljust(bar_width),
                ))
            log.hline()

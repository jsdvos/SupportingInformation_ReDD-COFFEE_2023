#!/usr/bin/env python
# -*- coding: utf-8 -*-

from molmod.periodic import periodic as pt
from molmod.units import *

import numpy as np

__all__ = ['LogFile']

class LogFile(object):
    def __init__(self, fn, restart = False):
        self.restart_fn = restart
        self.fn = fn
        self._read_coords()
        self.current_coords = self.coords[-1].copy()
    
    def _read_coords(self):
        self.numbers = None
        trajectory = []
        if self.restart_fn:
            f = open(self.fn.replace('_restart', ''), 'r')
        else:
            f = open(self.fn, 'r')
        lines = f.readlines()
        i = 0
        found = False
        resources, route, qz = self._get_com_header()
        sym = True
        for keyword in ['NoSymm', 'NoSymmetry', 'Symmetry=None']:
            if keyword in route:
                sym = False
        while i<len(lines):
            line = lines[i]
            if found:
                if line.lstrip(' ').startswith('--------------'):
                    found = False
                    if self.numbers is None:
                        self.numbers = np.array(numbers)
                    else:
                        assert (self.numbers==np.array(numbers)).all()
                    trajectory.append(coords)
                    i += 1
                else:
                    words = line.split()
                    numbers.append(int(words[1]))
                    coords.append(np.array([float(x)*angstrom for x in words[3:6]]))
                    i += 1
            elif 'orientation' in line:
                if (sym and line.lstrip(' ').startswith('Standard orientation:')) or (not sym and line.lstrip(' ').startswith('Input orientation:')):
                    found = True
                    numbers = []
                    coords = []
                    i += 5
                else:
                    i += 1
            else:
                i += 1
        f.close()
        self.coords = np.array(trajectory)

    def _get_com_header(self):
        resources = ''
        route = ''
        qz = ''
        if self.restart_fn:
            f = open(self.fn.replace('_restart', ''), 'r')
        else:
            f = open(self.fn, 'r')
        start_route = False
        end_route = False
        for line in f.readlines():
            line = line.lstrip(' ')
            words = line.split()
            if line.startswith('%nproc'):
                resources += line
            elif line.startswith('%mem'):
                resources += line
            elif line.startswith('%chk'):
                resources += line
            elif len(route)==0 and line.startswith('#'):
                start_route = True
            if start_route and not end_route:
                if not line.startswith('--------------'):
                    route += line.rstrip()
                else:
                    route += '\n'
                    end_route = True
            elif len(qz)==0 and line.startswith('Charge'):
                integers = []
                for word in words:
                    try:
                        integers.append(int(word))
                    except ValueError:
                        if '=' in word:
                            try:
                                integers.append(int(word.lstrip('=')))
                            except ValueError:
                                pass
                assert len(integers)==2, 'Could not find charge and multiplicity'
                charge, multiplicity = integers[0], integers[1]
                qz = '%i %i' %(charge, multiplicity)
        f.close()
        self.resources = resources
        self.route = route
        self.qz = qz
        return resources, route, qz

    def _read_freqs(self):
        freqs = []
        iramps = []
        modes = []
        f = open(self.fn, 'r')
        found = False
        for line in f.readlines():
            line = line.lstrip(' ')
            if line.startswith('Harmonic frequencies (cm**-1)'):
                found = True
                continue
            if found and line.startswith('-------------------'):
                found = False
                break
            if found:
                words = line.split()
                if len(words)==0: continue
                if words[0]=='Frequencies':
                    freqs.append(float(words[2])/centimeter)
                    freqs.append(float(words[3])/centimeter)
                    freqs.append(float(words[4])/centimeter)
                elif words[0]=='IR':
                    iramps.append(float(words[3])/centimeter)
                    iramps.append(float(words[4])/centimeter)
                    iramps.append(float(words[5])/centimeter)
                elif len(words)==3:
                    mode1 = np.zeros([len(self.numbers),3], float)
                    mode2 = np.zeros([len(self.numbers),3], float)
                    mode3 = np.zeros([len(self.numbers),3], float)
                else:
                    try:
                        iatom = int(words[0])-1
                    except ValueError:
                        continue
                    assert iatom>=0 and iatom<len(self.numbers)
                    assert int(words[1])==self.numbers[iatom]
                    mode1[iatom,:] = [float(word) for word in words[2:5]]
                    mode2[iatom,:] = [float(word) for word in words[5:8]]
                    mode3[iatom,:] = [float(word) for word in words[8:11]]
                    if iatom==len(self.numbers)-1:
                        modes.append(mode1)
                        modes.append(mode2)
                        modes.append(mode3)
        f.close()
        self.freqs = np.array(freqs)
        self.iramps = np.array(iramps)
        self.modes = np.array(modes)
        print('Found %i freqiencies for %i atoms'%(len(self.freqs), len(self.numbers)))
        assert self.freqs.shape==self.iramps.shape
        assert self.modes.shape==(len(self.freqs),len(self.numbers),3)

    def read_time(self):
        def get_time(line):
            tstamp = line.split(':')[1].split()
            assert len(tstamp) == 8
            assert tstamp[1] == 'days'
            assert tstamp[3] == 'hours'
            assert tstamp[5] == 'minutes'
            assert tstamp[7] == 'seconds.'
            minute = 60 # second
            hour = 60*minute
            day = 24*hour
            t = 0.0
            t += float(tstamp[0])*day
            t += float(tstamp[2])*hour
            t += float(tstamp[4])*minute
            t += float(tstamp[6])
            return t
        f = open(self.fn, 'r')
        cpu_time = None
        run_time = None
        for line in f.readlines():
            line = line.lstrip(' ')
            if line.startswith('Job cpu time:'):
                cpu_time = get_time(line)
            if line.startswith('Elapsed time:'):
                run_time = get_time(line)
        f.close()
        self.cpu_time = cpu_time # seconds
        self.run_time = run_time # seconds
        

    def perturb_neg_freq(self, amplitude=0.2*angstrom, n = 0):
        '''
            Perturb the geometry in the direction of the most negative
            frequency. Amplitude is the average displacement per atom made in
            the perturbation.
        '''
        if n > 0 and self.freqs[n]>0:
            print('n-th lowest frequency is positive, switching to lowest frequency')
            n = 0
        if self.freqs[0]>0:
            print('No negative frequencies found, no perturbation applied')
            return
        self.current_coords += amplitude*np.sqrt(len(self.numbers))*self.modes[n]
    
    def restart(self, fn_new, jobtype, lot = None, basis = None, disp=None, nosymm = False, chk=None, nproc=None, mem=None):
        resources, route, qz = self._get_com_header()
        data = route.strip().split()
        disp_added = False
        for i in range(len(data)):
            if '/' in data[i]:
                old_lot, old_basis = data[i].split('/')
                if lot == None:
                    lot = old_lot
                if basis == None:
                    basis = old_basis
                data[i] = '{}/{}'.format(lot, basis)
            if data[i].startswith('EmpiricalDispersion='):
                old_disp = data[i].split('=')[1]
                if disp == None:
                    disp = old_disp
                data[i] = 'EmpiricalDispersion=' + disp
                disp_added = True
        if nosymm and not 'NoSymm' in data:
            data.append('NoSymm')
        if not disp_added and not disp == None:
            data.append('EmpiricalDispersion=' + disp)
        route = ' '.join(data) + '\n'
        old_nproc, old_mem, old_chk = resources.split('%')[-3:]
        name, value = old_nproc.split('=')
        assert name == 'nproc'
        if nproc == None:
            nproc = int(value)
        name, value = old_mem.split('=')
        assert name == 'mem'
        if mem == None:
            assert value[-3:] == 'GB\n'
            mem = int(value[:-3])
        name, value = old_chk.split('=')
        if chk == None:
            chk = fn_new.replace('.com', '.chk').split('/')[-1]
        resources = '%nproc={}\n'.format(nproc)
        resources += '%mem={}GB\n'.format(mem)
        resources += '%chk={}\n'.format(chk)
        words = route.split()
        route_words = []
        jobtype_added = False
        for word in words:
            if word.split('(')[0] in ['opt', 'freq']:
                if not jobtype_added:
                    route_words.append(jobtype)
                    jobtype_added = True
            else:
                route_words.append(word)
        f = open(fn_new, 'w')
        print >> f, resources.rstrip('\n')
        print >> f, ' '.join(route_words)
        print >> f, ''
        print >> f, 'Restart from %s' %self.fn
        print >> f, ''
        print >> f, qz
        for number, coords in zip(self.numbers, self.current_coords):
            print >> f, '%2s  % .6f  % .6f  % .6f' %(pt[number].symbol, coords[0]/angstrom, coords[1]/angstrom, coords[2]/angstrom)
        print >> f, ''
        print >> f, ''
        print >> f, ""
        print >> f, ''
        print >> f, ''
        print >> f, ''
        f.close()
    

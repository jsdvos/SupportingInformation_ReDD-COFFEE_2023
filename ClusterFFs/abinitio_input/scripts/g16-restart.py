#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
from log import LogFile

def parse():
    usage = '%prog [options] gaussian.log'
    descr = '(Re)start new G09 job from previous job'
    parser = OptionParser(usage=usage, description=descr)
    parser.add_option(
        '-j', '--job', default='opt',
        help='The type of the new job. [default=%default]'
    )
    parser.add_option(
        '-l', '--lot', default = None,
        help='The level of theory of the new job [default = same as previous job]'
    )
    parser.add_option(
        '-b', '--basis', default = None,
        help='The basisset of the new job [default = same as previous job]'
    )
    parser.add_option(
        '-d', '--dispersion', default = None,
        help='Empirical Dispersion that is added [default = same as previous job]'
    )
    parser.add_option(
        '--nosymm', default = False, action='store_true',
        help='Turn symmetry off in the next job'
    )
    parser.add_option('-r', '--restart', default=False, action='store_true',
            help = "Indicates that the given log file is a restart file"
    )
    parser.add_option(
        '-s', '--suffix', default=None,
        help='A suffix added to the new job files with respect to the '+\
             'previous job files. By default a suffix -1 will be added '+\
             '(or -2 if -1 already in previous filename).'
    )
    parser.add_option(
        '-o', '--output', default=None,
        help='The name for the output file, this overrides the suffix option.'
    )
    parser.add_option(
        '-c', '--chk', default=None,
        help='The name of the chk file given in the new job'
    )
    parser.add_option(
        '-n', '--nproc', default=None,
        help='Number of processors needed for the new job [DEFAULT = same as previous job]'
    )
    parser.add_option(
        '-m', '--mem', default=None,
        help='Memory [in GB] needed for the new job [DEFAULT = same as previous job]'
    )
    parser.add_option(
        '-p', '--pert-negfreq', default=False, action='store_true',
        help='Perturb the geometry in the direction of the highest negative frequency'
    )
    options, args = parser.parse_args()
    assert len(args)==1, \
        'Only one argument expected, the log file of the previous G09 job.'
    fn_in = args[0]
    if options.output is not None:
        fn_out = options.output    
    elif options.suffix is None:
        if '-' in fn_in:
            pre, tmp = fn_in.split('-')
            i, suf = tmp.split('.')
            fn_out = '%s-%i.com' %(pre, int(i)+1)
        else:
            fn_out = fn_in.replace('.log', '-1.com')
    else:
        fn_out = fn_in.replace('.log', '%s.com' %options.suffix)
    return fn_in, fn_out, options

def main():
    fn_in, fn_out, options = parse()
    log = LogFile(fn_in, restart = options.restart)
    if options.pert_negfreq:
        log._read_freqs()
        log.perturb_neg_freq(n = 2)
    log.restart(fn_out, options.job, lot = options.lot, basis = options.basis,
            disp = options.dispersion, nosymm = options.nosymm, chk = options.chk,
            nproc = options.nproc, mem = options.mem)

if __name__=='__main__':
    main()

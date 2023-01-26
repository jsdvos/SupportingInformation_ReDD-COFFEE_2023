#! /usr/bin/env python

import numpy as np
from optparse import OptionParser

from molmod.periodic import periodic
from molmod.io import dump_chk, load_chk
from yaff.system import System
from yaff.log import log
log.set_level(0)

def parse():
    usage = '%prog [options] system.chk ffatypes.txt'
    descr = 'Create new CHK file with assigned ffatypes, as defined in ffatypes.txt'
    parser = OptionParser(usage = usage, description = descr)
    parser.add_option(
            '-o', '--output', default = None,
            help = 'The name of the output System CHK file [default = {system}_ffatype.chk]')
    options, args = parser.parse_args()
    if not len(args) == 2 or not args[0].endswith('.chk') or not args[1].endswith('.txt'):
        raise ValueError('Exactly two arguments expected: the System file (CHK) and ffatype definitions TXT file')
    fn_sys, fn_ffatypes = args
    if options.output == None:
        fn_out = fn_sys.replace('.chk', '_ffatype.chk')
    else:
        fn_out = options.output
    return fn_sys, fn_out, fn_ffatypes

def get_ffatypes(fn_ffatypes, sys):
    ffatypes = []
    with open(fn_ffatypes, 'r') as f:
        for i, line in enumerate(f.readlines()):
            ffatype = line.strip()
            data = ffatype.split('_')
            symbol = ''
            for char in data[0]:
                if not char.isdigit():
                    symbol += char
            assert symbol == periodic[sys.numbers[i]].symbol
            ffatypes.append(ffatype)
    return np.array(ffatypes)

def main():
    fn_sys, fn_out, fn_ffatypes = parse()
    
    # Create System
    sys = System.from_file(fn_sys)
    sys.detect_bonds()
    print('System created from file {}'.format(fn_sys))

    # Read ffatypes
    ffatypes = get_ffatypes(fn_ffatypes, sys)
    print('FFatypes defined')

    # Write System
    temp = load_chk(fn_sys)
    temp['ffatypes'] = ffatypes
    dump_chk(fn_out, temp)
    print('New CHK file created with ffatypes to file {}'.format(fn_out))

if __name__ == '__main__':
    main()


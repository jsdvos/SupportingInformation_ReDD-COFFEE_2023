#! /usr/bin/env python
import sys
import os

from molmod.molecular_graphs import MolecularGraph
from molmod.periodic import periodic
from molmod.io import dump_chk, load_chk
from yaff.system import System
from yaff.log import log
log.set_level(0)

from collections import Counter

from optparse import OptionParser

def parse():
    usage = '%prog [options] system.chk definitions.py'
    descr = 'Define the atom types for a System, given some filters'
    parser = OptionParser(usage = usage, description = descr)
    parser.add_option(
            '-o', '--output', default = None,
            help = 'The name of the output System CHK file [default = {system}_ffatype.chk]'
    )
    options, args = parser.parse_args()
    if not len(args) == 2 or not args[0].endswith('.chk') or not args[1].endswith('.py'):
        raise ValueError('Exactly two arguments expected: the System file (CHK) and ffatype definition PY script')
    fn_sys = args[0]
    if options.output == None:
        fn_out = fn_sys.replace('.chk', '_ffatype.chk')
    else:
        fn_out = options.output

    # Load afilters
    path = os.path.dirname(os.path.abspath(args[1]))
    if '/' in args[1]:
        fn_def = args[1].rsplit('/', 1)[0]
    else:
        fn_def = args[1]
    fn_def, ext = os.path.splitext(fn_def)
    if not path == '':
        sys.path.insert(1, path)
    exec('from {} import afilters'.format(fn_def), globals())
    return fn_sys, fn_out, afilters

def get_ffatypes(system, afilters):
    # Create Molecular Graph
    graph = MolecularGraph(system.bonds, system.numbers)

    # Get ffatypes
    ffatypes = [-1]*len(system.numbers)
    for ffatype, afilter in afilters:
        print(ffatype)
        for iatom, number in enumerate(system.numbers):
            if afilter(iatom, graph) and ffatypes[iatom] == -1:
                ffatypes[iatom] = ffatype
                print('{:21s}: {}'.format(ffatype, iatom))
    for iatom, number in enumerate(system.numbers):
        if ffatypes[iatom] == -1:
            symbol = periodic[system.numbers[iatom]].symbol
            print('No atom type found for atom {} ({}) with neighbors:'.format(iatom, symbol))
            for neighbor in graph.neighbors[iatom]:
                neighbor_ffatype = ffatypes[neighbor]
                if neighbor_ffatype == -1:
                    print('    {} ({})'.format(neighbor, periodic[system.numbers[neighbor]].symbol))
                else:
                    print('    {} ({})'.format(neighbor, neighbor_ffatype))
    print('\n')

    # Print summary
    list_ffatypes = dict(Counter(ffatypes))
    print('Number of occurences per atom type:')
    for key in list_ffatypes.keys():
        try:
            print('{:21s}: {}'.format(key, list_ffatypes[key]))
        except ValueError:
            print('An error occured for atom type {}'.format(key))
    
    return ffatypes

def main():
    fn_sys, fn_out, afilters = parse()

    # Create System
    system = System.from_file(fn_sys)
    system.detect_bonds()
    print('System created from file {}'.format(fn_sys))

    # Get ffatypes
    ffatypes = get_ffatypes(system, afilters)
    print('FFatypes defined')


    # Create new chk file with ffatypes
    temp = load_chk(fn_sys)
    temp['ffatypes'] = ffatypes
    dump_chk(fn_out, temp)
    print('New CHK file created with ffatypes to file {}'.format(fn_out))


if __name__ == '__main__':
    main()


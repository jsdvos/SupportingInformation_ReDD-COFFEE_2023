import os

from optparse import OptionParser

from yaff import System

from uff import UFFMachine

def parse():
    usage = 'python %prog [options] sys'
    descr = 'Create a UFF force field for the given system'
    parser = OptionParser(usage = usage, description = descr)
    parser.add_option(
            '-f', '--ffatypes', default = None,
            help = 'File to check and correct the UFF ffatypes'
    )
    parser.add_option(
            '-b', '--bond_orders', default = None,
            help = 'File to check and correct the bond orders'
    )
    parser.add_option(
            '-c', '--cov', default = None,
            help = 'Parameter file of the covalent parameters [if not given: no output]'
    )
    parser.add_option(
            '-l', '--lj', default = None,
            help = 'Parameter file of the LJ parameters [if not given: no output]'
    )
    options, args = parser.parse_args()
    assert len(args) == 1 and args[0].split('.')[-1] == 'chk', 'Exactly one argument expected: input CHK system (got {})'.format(args)
    fn_sys = args[0]
    if options.ffatypes == None:
        options.ffatypes = fn_sys.replace('.chk', '_ffatypes.txt')
    if options.bond_orders == None:
        options.bond_orders == fn_sys.replace('.chk', '_bonds.txt')
    if options.cov == None and options.lj == None:
        print('WARNING: No parameters will be written out, should specifcy options (-c/--cov or -l/--lj)')
    return fn_sys, options.ffatypes, options.bond_orders, options.cov, options.lj

if __name__ == '__main__':
    fn_sys, fn_ffatypes, fn_bonds, fn_cov, fn_lj = parse()
    sys = System.from_file(fn_sys)
    print('Composing UFF for system {}'.format(fn_sys))
    uff = UFFMachine(sys)
    if os.path.exists(fn_ffatypes):
        uff.read_uff_ffatypes(fn_ffatypes)
        print('Read UFF ffatypes from {}'.format(fn_ffatypes))
    else:
        uff.write_uff_ffatypes(fn_ffatypes)
        print('UFF ffatypes written to {}'.format(fn_ffatypes))
    if os.path.exists(fn_bonds):
        uff.read_bond_orders(fn_bonds)
        print('Read bond orders from {}'.format(fn_bonds))
    else:
        uff.write_bond_orders(fn_bonds)
        print('Bond orders written to {}'.format(fn_bonds))
    uff.build()
    if not fn_cov == None:
        uff.pars_cov.to_file(fn_cov)
        print('Covalent part written to {}'.format(fn_cov))
    if not fn_lj == None:
        uff.pars_lj.to_file(fn_lj)
        print('LJ part written to {}'.format(fn_lj))

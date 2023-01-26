#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
from yaff import System, angstrom, log
from molmod.periodic import periodic


def parse():
    usage = 'python %prog [options] geometry.xyz'
    descr = 'Make Gaussian com file from input geometry in xyz-format'
    parser = OptionParser(usage=usage, description=descr)
    parser.add_option(
        '-o', '--output', default = None, help='The name of the output com-file'
    )
    parser.add_option(
        '--mem', default='70', help='Memory (in GB) [default=%default]'
    )
    parser.add_option(
        '-n', '--nproc', default=15, help='Number of processors [default=%default]'
    )
    parser.add_option(
        '-c', '--charge', default = 0, help='Total charge of the system [default=%default]'
    )
    parser.add_option(
        '-m', '--multiplicity', default = 1, help='Multiplicity of the system [default=%default]'
    )
    parser.add_option(
        '-j', '--job', default = 'opt', help='Job type: opt or freq [default=%default]'
    )
    parser.add_option(
        '-b', '--basis', default = '6-311++G(d,p)', help='Basis set [default=%default]'
    )
    parser.add_option(
        '-l', '--lot', default = 'B3LYP', help='Level of Theory [default=%default]'
    )
    parser.add_option(
        '-d', '--dispersion', default = None,
        help='Empirical Dispersion correction [Examples: GD3 (Grimme D3)]'
    )
    parser.add_option(
        '-q', '--quiet', default = False,
        action = 'store_true', help = 'No yaff output is given'
    )
    options, args = parser.parse_args()
    assert len(args) == 1, \
        'Only one argument expected: input geometry'
    
    fn_in = args[0]
    if options.output is not None:
        fn_out = options.output
    else:
        fn_out = fn_in.replace('.xyz', '.com')
    if options.quiet:
        log.set_level(0)
    return fn_in, fn_out, options
    

def main():
    fn_in, fn_out, options = parse()
    print(fn_in)
    sys = System.from_file(fn_in)
    with open(fn_out, 'w') as f:
        f.write('%nproc={}\n'.format(options.nproc))
        f.write('%mem={}GB\n'.format(options.mem))
        f.write('%chk={}\n'.format(fn_out.replace('.com', '.chk').split('/')[-1]))
        if options.dispersion == None:
            f.write('#P {}/{} {} SCF=YQC NoSymm\n'.format(options.lot, options.basis, options.job))
        else:
            f.write('#P {}/{} EmpiricalDispersion={} {} SCF=YQC NoSymm\n'.format(options.lot, options.basis, options.dispersion, options.job))
        f.write('\n')
        f.write('Comment\n')
        f.write('\n')
        f.write('{} {}\n'.format(options.charge, options.multiplicity))
        for i in range(sys.natom):
            element = periodic[sys.numbers[i]].symbol
            pos = sys.pos[i]/angstrom
            if len(element) == 1:
                element = ' ' + element
            f.write('{}    {:.15f}    {:.15f}    {:.15f}\n'.format(element, pos[0], pos[1], pos[2]))
        f.write('\n\n\n\n')
            
if __name__=='__main__':
    main()    

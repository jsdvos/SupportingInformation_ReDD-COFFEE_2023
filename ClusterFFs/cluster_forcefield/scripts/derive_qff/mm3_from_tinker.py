import os
from optparse import OptionParser
import numpy as np
from yaff import System, angstrom, log
from molmod.periodic import periodic
from mm3 import mm3

log.set_level(0)

def parse():
    usage = '%prog [options] system.chk system_tinker_mm3.xyz'
    descr = 'Obtain a yaff Parameters file for the MM3 VdW parameters of a specific System using the Molden tinker MM3 feature'
    parser = OptionParser(usage = usage, description = descr)
    parser.add_option(
        '-o', '--output', default=None,
        help = 'The name of the output Parameter file'
    )
    options, args = parser.parse_args()
    if not len(args) == 2 or not args[0].endswith('.chk') or not args[1].endswith('.xyz'):
        raise ValueError('Exactly two arguments expected, the CHK System file and the XYZ Tinker file')
    fn_sys, fn_xyz = args
    if options.output == None:
        if '/' in fn_sys:
            path = fn_sys.rsplit('/', 1)[0]
            fn_out = os.path.join(path, 'pars_mm3.txt')
        else:
            fn_out = 'pars_mm3.txt'
    else:
        fn_out = options.output
    return fn_sys, fn_xyz, fn_out

def get_mm3_indices(fn_sys, fn_xyz):
    sys = System.from_file(fn_sys)
    mm3_ffatypes = {ffatype: None for ffatype in sys.ffatypes}
    with open(fn_xyz, 'r') as f:
        data = f.readline().strip().split()
        assert int(data[0]) == sys.natom
        for i in range(sys.natom):
            data = f.readline().strip().split()
            assert i+1 == int(data[0])
            symbol = str(data[1])
            x = float(data[2])*angstrom
            y = float(data[3])*angstrom
            z = float(data[4])*angstrom
            pos = np.array([x, y, z])
            mm3_index = int(data[5])
            for j in range(sys.natom):
                if np.linalg.norm(pos - sys.pos[j]) < 1e-4:
                    assert periodic[symbol].number == sys.numbers[j]
                    ffatype = sys.get_ffatype(j)
                    if mm3_ffatypes.get(ffatype) == None:
                        mm3_ffatypes[ffatype] = mm3_index
                    else:
                        if not mm3_ffatypes.get(ffatype) == mm3_index:
                            raise RuntimeError('Different MM3 parameters recognized for ffatype {}'.format(ffatype))
    if '/' in fn_sys:
        sys_name = fn_sys.split('/', 1)[1]
    else:
        sys_name = fn_sys
    if None in mm3_ffatypes.values():
        print('Could not find MM3 ffatype for following atoms:')
        for ffatype, mm3_index in mm3_ffatypes.items():
            if mm3_index is None:
                print('\t{}'.format(ffatype))
    else:
        print('MM3 ffatypes for System {} found'.format(sys_name))
    return mm3_ffatypes

def write_mm3_pars(mm3_ffatypes, fn_out):
    with open(fn_out, 'w') as f:
        f.write('MM3:UNIT SIGMA angstrom\n')
        f.write('MM3:UNIT EPSILON kcalmol\n')
        f.write('MM3:SCALE 1 0.0\n')
        f.write('MM3:SCALE 2 0.0\n')
        f.write('MM3:SCALE 3 1.0\n')
        f.write('\n\n')
        for ffatype, mm3_index in mm3_ffatypes.items():
            sigma, epsilon = mm3[mm3_index]
            f.write('MM3:PARS {} {} {} 0 \n'.format(ffatype, sigma, epsilon))
    print('MM3 parameters written to {}'.format(fn_out))

def main():
    fn_sys, fn_xyz, fn_out = parse()
    mm3_ffatypes = get_mm3_indices(fn_sys, fn_xyz)
    write_mm3_pars(mm3_ffatypes, fn_out)

if __name__ == '__main__':
    main()


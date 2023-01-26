import sys

from datetime import datetime

import dask

import yaff.log
yaff.log.set_level(0)

from core import StructureOptimization

###### UTIL FUNCTIONS ######
def run_one(name, only_yaff = False):
    # WORK FUNCTION
    try:
        print('{} started at {}'.format(name, datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
        sys.stdout.flush()
        optimizer = StructureOptimization(name, only_yaff = only_yaff)
        optimizer.run()
        print('{} ended at {}'.format(name, datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
    except Exception as e:
        print('ERROR: {} crashed due to following error:\n{}'.format(name, e))
    sys.stdout.flush()

def run_all(fn_input, only_yaff = False, run_dask = True):
    opts = []
    with open(fn_input, 'r') as f:
        for count, line in enumerate(f):
            words = line.strip().split()
            if len(words) == 0: continue
            if words[0][0] == "#": continue
            folder = words[0]
            name = words[1]
            if len(words) > 2:
                if words[2] == 'only_yaff':
                    only_yaff_temp = True
                else:
                    print('{} (skipping line {}): dont know what to do with argument {}, should be {}'.format(name, count + 1, words[2], 'only_yaff'))
                    continue
            else:
                only_yaff_temp = only_yaff
            if run_dask:
                run = dask.delayed(run_one)
                opts.append(run(name, only_yaff = only_yaff_temp))
            else:
                run_one(name, only_yaff = only_yaff_temp)
    if run_dask:
        opts = dask.compute(*opts)

if __name__ == '__main__':
    # Setup run
    args = sys.argv
    fn_input = args[1]
    only_yaff = False
    if len(args) > 2:
        if args[2] == 'only_yaff':
            only_yaff = True
            print('LAMMPS force field turned OFF')
        else:
            print('Dont know what to do with arguments (got: {})'.format(args))
            print('Continue with {} as fn_input and without only_yaff'.format(fn_input))
    
    run_dask = (not fn_input == 'input_single.txt')
    if run_dask:
        # Startup dask
        from dask.distributed import Client, progress
        client = Client() # take max processes, threads and mem

    # Run
    run_all(fn_input, only_yaff = only_yaff, run_dask = run_dask)

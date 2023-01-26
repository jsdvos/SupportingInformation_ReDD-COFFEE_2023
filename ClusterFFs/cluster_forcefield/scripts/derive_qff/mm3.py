import os
from socket import gethostname

def get_mm3():
    '''
    Load all MM3 VdW interaction parameters

    Returns a dictionary that maps the index of the MM3 number onto the epsilon and sigma values
    '''
    mm3 = {}
    with open('mm3.prm', 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) > 0 and  data[0] == 'vdw':
                mm3_number = int(data[1])
                sigma = float(data[2])
                epsilon = float(data[3])
                mm3[mm3_number] = [sigma, epsilon]
    print('WARNING: the units of the MM3 parameters in the mm3 dictionary are not atomic units, but kcalmol (epsilon) and angstrom (sigma)')
    return mm3

mm3 = get_mm3()

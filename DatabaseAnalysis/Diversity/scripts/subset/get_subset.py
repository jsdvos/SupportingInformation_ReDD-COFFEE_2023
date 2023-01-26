import os

import numpy as np
import pandas as pd
import scipy
import scipy.spatial

# STEP 1: Read data from csv file
def get_data():
    result = pd.read_csv('../../data/features.txt', sep = ';')
    return result, result['database'].values

# STEP 2: Process data and extract parameters that you want to use
def process_data(data, domain):
    # STEP 2a: create additional data
    data['mass'] = data['rho']*data['vol']
    for feature in ['asa', 'av']:
        data[feature + '_m'] = data[feature]/data['mass']
        data[feature + '_v'] = data[feature]/data['vol']

    # STEP 2b: extract wanted features
    geometry_features = ['di', 'df', 'dif', 'rho', 'asa_m', 'asa_v', 'av_m', 'av_v']
    rac_features = ['prod_I_0', 'prod_I_1', 'prod_I_2', 'prod_I_3', 'prod_T_0', 'prod_T_1', 'prod_T_2', 'prod_T_3', 'prod_X_0', 'prod_X_1', 'prod_X_2', 'prod_X_3', 'prod_S_0', 'prod_S_1', 'prod_S_2', 'prod_S_3', 'prod_Z_0', 'prod_Z_1', 'prod_Z_2', 'prod_Z_3', 'prod_a_0', 'prod_a_1', 'prod_a_2', 'prod_a_3', 'diff_I_0', 'diff_I_1', 'diff_I_2', 'diff_I_3', 'diff_T_0', 'diff_T_1', 'diff_T_2', 'diff_T_3', 'diff_X_0', 'diff_X_1', 'diff_X_2', 'diff_X_3', 'diff_S_0', 'diff_S_1', 'diff_S_2', 'diff_S_3', 'diff_Z_0', 'diff_Z_1', 'diff_Z_2', 'diff_Z_3', 'diff_a_0', 'diff_a_1', 'diff_a_2', 'diff_a_3']
    ligand_features = ['lig_' + feat for feat in rac_features]
    linker_features = ['link_' + feat for feat in rac_features]
    functional_features = ['func_' + feat for feat in rac_features]
    
    features = {'geometry': geometry_features,
            'linkage': ligand_features,
            'linker': linker_features,
            'functional': functional_features,
            'all': geometry_features + ligand_features + linker_features + functional_features
            }[domain]
    features = [feature for feature in features if not np.std(data[feature]) == 0]
    x_data = data[features].values
    
    # Normalize data
    x_mu = np.mean(x_data, axis = 0)
    x_std = np.std(x_data, axis = 0)
    x_data = (x_data - x_mu)/x_std
    return x_data

# Reduce the database by selecting a varied set of structures
def maxmin(nsamples, data, names, databases, start = 0, fn_out = 'selected_ids.txt', nwrite = 100):
    selected_ids = [start] # COF-5
    selected_data = [data[i] for i in selected_ids]
    while len(selected_ids) < nsamples:
        last_entry = selected_data[-1].reshape(1, -1)
        dtemp = scipy.spatial.distance.cdist(last_entry, data, metric = "euclidean")
        d = np.asarray(dtemp)
        if len(selected_data) > 1:
            dmat_small = np.vstack([dmat_small, d])
        else:
            dmat_small = d
        dmat_small = np.amin(dmat_small, axis = 0).reshape(1, -1)
        new_ind = np.argmax(dmat_small)
        selected_ids.append(new_ind)
        selected_data.append(data[new_ind])
        if len(selected_ids) % nwrite == 0 or len(selected_ids) == nsamples:
            with open(fn_out, 'w') as f:
                for i in selected_ids:
                    f.write('{} {} {}\n'.format(i, databases[i], names[i]))


if __name__ == '__main__':
    def get_ids(database, databases, nstruct = 10000):
        ids = []
        if database == 'subset':
            with open('../../data/subset/sorted_maxmin.txt', 'r') as f:
                for line in f.readlines()[:nstruct]:
                    struct_id, link, struct = line.strip().split()
                    ids.append(int(struct_id))
        else:
            for i, label in enumerate(databases):
                if database == label:
                    ids.append(i)
        return ids

    data, databases = get_data()
    data = data[data['database'] == 'redd-coffee']
    databases = [databases[i] for i in range(len(databases)) if databases[i] == 'redd-coffee']
    if not data['struct'].values[337] == 'hcb_18-08-01_01-01-01':
        print('WARNING: COF-5 has changed id')
    x_data = process_data(data, 'all')
    maxmin(len(x_data), x_data, data['struct'].values, databases, start = 337, fn_out = '../../data/subset/sorted_maxmin.txt', nwrite = 1000)



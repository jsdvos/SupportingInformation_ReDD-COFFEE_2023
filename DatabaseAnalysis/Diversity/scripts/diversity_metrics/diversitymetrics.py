import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from shapely.geometry import *

from alpha import alpha_shape

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
            }[domain]
    features = [feature for feature in features if not np.std(data[feature]) == 0]
    x_data = data[features].values
    
    # Normalize data
    x_mu = np.mean(x_data, axis = 0)
    x_std = np.std(x_data, axis = 0)
    x_data = (x_data - x_mu)/x_std
    return x_data

# STEP 3: Calculate Variety (V), Balance (B) and Disparity (D)
def get_bins(data, domain, n_clusters = 1000):
    # Step 3a: partition material space in bins
    fn = '../../data/diversity_metrics/{}.bin'.format(domain)
    bins = []
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            for line in f:
                bins.append(int(line.strip()))
        print('Read bins from ' + fn)
    else:
        # k-means clustering
        kmeans = KMeans(n_clusters = n_clusters, n_init = 1)
        kmeans.fit(data)
        bins = kmeans.labels_
        with open(fn, 'w') as f:
            for b in bins:
                f.write('{}\n'.format(b))
    return bins


def get_variety(bins, ids):
    # Variety (V): number of bins that are occupied by given set of structures
    tot_bins = max(bins) + 1
    count_bins = set([])
    for i in ids:
        count_bins.update([bins[i]])
    return float(len(count_bins))/tot_bins

def get_pielou(bins, ids):
    # Balance (B)
    p = {x: 0 for x in range(max(bins) + 1)}
    for i in ids:
        p[bins[i]] += 1
    n = len([x for x in p.keys() if not p[x] == 0])
    tot_structs = sum(p.values())
    h_max = -np.log(1.0/n)
    h = 0.0
    kl = 0.0
    for x, value in p.items():
        if value == 0: continue
        value = float(value)/tot_structs
        h += -value*np.log(value)
        kl += value*np.log(value*n)
    pielou = (1-np.exp(h))/(1-np.exp(h_max))
    return pielou

def get_concave_area(data):
    points = [Point(coords) for coords in data]
    hull, edges = alpha_shape(points, 1.0)
    area = hull.area
    result = [area]
    return tuple(result)

def get_disparity(data, ids):
    # Disparity (D)
    pca = PCA(n_components = 2)
    x_all = pca.fit_transform(data)
    x = [x_all[i] for i in ids]
    area_x = get_concave_area(x)[0]
    area_x_all = get_concave_area(x_all)[0]
    return area_x/area_x_all

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

    projects = ['diversity_metrics', 'subset_size']

    # Diversity metrics
    if 'diversity_metrics' in projects:
        result = {}
        data, databases = get_data()
        for domain in ['geometry', 'linker', 'linkage', 'functional']:
            result[domain] = {}
            x_data = process_data(data, domain)
            bins = get_bins(x_data, domain)
            for database in ['redd-coffee', 'core', 'curated', 'martin', 'mercado', 'subset']:
                ids = get_ids(database, databases)
                result[domain][database + '_v'] = get_variety(bins, ids)
                result[domain][database + '_b'] = get_pielou(bins, ids)
                result[domain][database + '_d'] = get_disparity(x_data, ids)
        # Write out
        fn_out = '../../data/diversity_metrics/diversity_metrics.dat'
        with open(fn_out, 'w') as f:
            f.write('{:10s} '.format('Domain'))
            for database in ['redd', 'core', 'curated', 'martin', 'mercado', 'subset']:
                f.write('{:10s} {:10s} {:10s} '.format(database + '_v', database + '_b', database + '_d'))
            f.write('\n')
            for domain in ['geometry', 'linker', 'linkage', 'functional']:
                f.write('{:10} '.format(domain))
                for database in ['redd-coffee', 'core', 'curated', 'martin', 'mercado', 'subset']:
                    v = result[domain][database + '_v']
                    b = result[domain][database + '_b']
                    d = result[domain][database + '_d']
                    f.write('{:10} {:10} {:10} '.format(v, b, d))
                f.write('\n')

    # Evolution of V, B, D when subset size grows
    if 'subset_size' in projects:
        data, databases = get_data()
        subset_ids = get_ids('subset', databases, nstruct = 20000)
        for domain in ['geometry', 'linker', 'linkage', 'functional']:
            x_data = process_data(data, domain)
            bins = get_bins(x_data, domain)
            for i in range(100, 15001, 100):
                v = get_variety(bins, subset_ids[:i])
                b = get_pielou(bins, subset_ids[:i])
                d = get_disparity(x_data, subset_ids[:i])
                with open('../../data/diversity_metrics/vbd_subset_size.txt', 'a') as f:
                    f.write('{} {} {} {} {}\n'.format(domain, i, v, b, d))





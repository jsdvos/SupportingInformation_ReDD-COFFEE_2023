import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import *
import scipy
from collections import Counter

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


def get_bins(data, domain, n_clusters = 1000):
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

def get_concave_area(data, get_edge = False, get_hull = False):
    points = [Point(coords) for coords in data]
    hull, edges = alpha_shape(points, 1.0)
    area = hull.area
    result = [area]
    if get_edge:
        result.append(edges)
    if get_hull:
        result.append(hull)
    return tuple(result)

def plot_bin_distribution(x_data, bins, databases, fn_fig):
    color_labels = {
            'redd-coffee': ((183, 21, 22), 'ReDD-COFFEE'),
            'subset': ((239, 118, 119), 'Subset'),
            'core': ((47, 108, 157), 'CoRE'),
            'curated': ((137, 183, 220), 'CURATED'),
            'martin': ((63, 143, 61), 'Martin'),
            'mercado': ((149, 210, 147), 'Mercado')
            }
    for zorder, database in enumerate(['redd-coffee', 'subset', 'core', 'curated', 'martin', 'mercado']):
        ids = get_ids(database, databases)
        bins_db = [bins[i] for i in ids]
        count = Counter(bins_db)
        color, label = color_labels[database]
        color = [float(c)/255 for c in color]
        plt.plot(np.linspace(0, 1, len(count)), np.array(sorted(count.values()), float)/sum(count.values()), label = label, color = color, zorder = -zorder)
    plt.yscale('log')
    plt.xlabel('Rank order of bins')
    plt.ylabel('Proportional abundance')
    plt.legend()
    plt.savefig(fn_fig, bbox_inches = 'tight')
    plt.clf()

def ring_coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes

def pathify(polygon):
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = np.concatenate(
                    [np.asarray(polygon.exterior)]
                    + [np.asarray(r) for r in polygon.interiors])
    codes = np.concatenate(
                [ring_coding(polygon.exterior)]
                + [ring_coding(r) for r in polygon.interiors])
    return Path(vertices, codes)

def plot(x_data, db_ids, plot = 'pca', fn_save = None, fn_out = None):
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    # Fit model and save data if asked
    if fn_save is None or not os.path.exists(fn_save):
        if plot == 'pca':
            model = PCA(n_components = 2)
        elif plot == 'tsne':
            model = TSNE(n_components = 2, init = 'pca', random_state = 1891)
        data = model.fit_transform(x_data)
        if fn_save is not None:
            with open(fn_save, 'w') as f:
                for i in range(len(data)):
                    f.write('{} {}\n'.format(data[i][0], data[i][1]))
    else:
        data = []
        with open(fn_save, 'r') as f:
            for line in f.readlines():
                x, y = line.strip().split()
                data.append([float(x), float(y)])
        assert len(data) == len(x_data), 'Not all data is present in ' + fn_save
    x_all = [data[i][0] for i in range(len(data))]
    y_all = [data[i][1] for i in range(len(data))]

    fig, ax = plt.subplots()
    # Layer 1: Gray scatter plot with all data points
    ax.scatter(x_all, y_all, c = 'gray', s = 12, zorder = 1, rasterized = True)
    if plot == 'pca':
        # Layer 2: Concave hull of own database
        points = [Point([x_all[i], y_all[i]]) for i in db_ids]
        hull, edges = alpha_shape(points, 1.0)
        if hull.type == 'Polygon':
            polygons = [hull]
        else:
            polygons = hull
        for polygon in polygons:
            path = pathify(polygon)
            patch = PathPatch(path, fc = (0.0, 0.0, 0.0, 0.0), ec = 'k', lw = 3, zorder = 2)
            ax.add_patch(patch)
    # Layer 3: density of subset data points
    x = [x_all[i] for i in db_ids]
    y = [y_all[i] for i in db_ids]
    cmap = ax.hexbin(x, y, cmap = 'YlOrRd', mincnt = 1, bins = 'log', zorder = 3, rasterized = True)
    plt.axis('off')
    colorbar = fig.colorbar(cmap, ax = ax, orientation = 'horizontal', aspect = 40, pad = 0.0)
    plt.savefig(fn_out, bbox_inches = 'tight', dpi = 200)



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

    projects = ['bin_distribution', 'pca_tsne']

    # Bin distribution
    if 'bin_distribution' in projects:
        data, databases = get_data()
        for domain in ['geometry', 'linker', 'linkage', 'functional']:
            x_data = process_data(data, domain)
            bins = get_bins(x_data, domain)
            plot_bin_distribution(x_data, bins, databases, fn_fig = '../../figs/diversity_metrics/bins_{}.pdf'.format(domain))

    # Plot with concave hull and heat map
    if 'pca_tsne' in projects:
        data, databases = get_data()
        for db in ['core', 'curated', 'martin', 'mercado', 'redd-coffee', 'subset']:
            db_ids = get_ids(db, databases)
            for domain in ['geometry', 'linker', 'linkage', 'functional']:
                x_data = process_data(data, domain)
                for model in ['pca', 'tsne']: # Try also with pca
                    fn_save = '../../data/diversity_metrics/{}_{}.dat'.format(domain, model)
                    fn_out = '../../figs/diversity_metrics/{}_{}_{}.pdf'.format(domain, model, db)
                    plot(x_data, db_ids, plot = model, fn_save = fn_save, fn_out = fn_out)


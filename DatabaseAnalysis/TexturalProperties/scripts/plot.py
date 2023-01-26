import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib import cm
import scipy.stats as stats
import random
import time

from molmod.units import kilogram, gram, nanometer, centimeter, meter

def get_data(databases = None):
    result = pd.DataFrame()
    result = pd.read_csv('../data/structural.csv', sep = ';')
    result['mass'] = result['rho']*result['vol']
    for feature in ['asa', 'av', 'nasa', 'nav']:
        result[feature + '_m'] = result[feature]/result['mass']
        result[feature + '_v'] = result[feature]/result['vol']
    return result

def extract(data, database, dim = None, linkage = None):
    db_data = data[data['database'] == database]
    if dim is not None:
        if not database == 'redd-coffee':
            print('Dont know how to define dimensionality in ' + database)
            print('Continue with full database')
            return db_data
        else:
            structs = db_data['struct'].values
            tops_2d = os.listdir('../../../DatabaseGeneration/StructureAssembly/data/Topologies/2D')
            flags_2d = [struct.split('_')[0] + '.top' in tops_2d for struct in structs]
            if dim == 2:
                return db_data[flags_2d]
            else:
                return db_data[[not flag for flag in flags_2d]]
    if linkage is not None:
        if not database == 'redd-coffee':
            print('Dont know how to define linkages in ' + database)
            print('Continue with full database')
            return db_data
        else:
            try:
                linkage = int(linkage)
            except:
                linkage_id = {'Boronate Ester': 1, 'Boroxine': 2, 'Borosilicate': 3, 'Imine': 4, '(Acyl)hydrazone': 5, 'Azine': 6, 'Oxazoline': 8, '(Keto)enamine': 9, 'Enamine': 9, 'Ketoenamine': 9, 'Triazine': 10, 'Borazine': 11, 'Imide': 12}[linkage]
            linkage_ids = []
            for struct in db_data['struct'].values:
                for sbu in struct.split('_')[1:]:
                    if '-' in sbu:
                        linkage_ids.append(int(sbu.split('-')[-1]))
                        break
            assert len(linkage_ids) == len(db_data)
            mask = [struct_linkage_id == linkage_id for struct_linkage_id in linkage_ids]
            return db_data[mask]
    return db_data
        

def plot_x_kernel(ax, x, c, label = None, N = 1000):
    kde = stats.gaussian_kde(x)
    kde.covariance_factor = lambda: 0.25
    kde._compute_covariance()
    xmin, xmax = min(x), max(x)
    xgrid = np.linspace(xmin, xmax, N)
    z = kde(xgrid)
    return ax.plot(xgrid, z, c = c, label = label)

def plot_xy_kernel(ax, x, y, cmap, N = 100, rasterized = False):
    data = np.vstack([x, y])
    kde = stats.gaussian_kde(data)
    xgrid = np.linspace(min(x), max(x), N)
    ygrid = np.linspace(min(y), max(y), N)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    dx = (xgrid[1] - xgrid[0])/2.
    dy = (ygrid[1] - ygrid[0])/2.
    extent = (xgrid[0] - dx, xgrid[-1] + dx, ygrid[0] - dy, ygrid[-1] + dy)
    z[z < np.max(z)*1e-2] = np.nan
    return ax.imshow(z.reshape(Xgrid.shape), cmap = cmap, origin = 'lower', extent = extent, rasterized = rasterized)


def create_colormap(c, N = 256, delta = 0.0, min_opacity = 0.0):
    vals = np.ones((N, 4))
    r, g, b = colors.to_rgb(c)
    r0 = (1-r*delta)/(1-delta)
    g0 = (1-g*delta)/(1-delta)
    b0 = (1-b*delta)/(1-delta)
    vals[:, 0] = np.linspace(r0, r, N)
    vals[:, 1] = np.linspace(g0, g, N)
    vals[:, 2] = np.linspace(b0, b, N)
    vals[:, 3] = np.linspace(min_opacity, 1, N)
    return colors.ListedColormap(vals)


if __name__ == '__main__':
    databases = ['hmof', 'qmof', 'tobacco', 'iza', 'redd-coffee']
    database_colors = {
            'redd-coffee': (183.0/255, 21.0/255, 22.0/255),
            'iza': (129.0/255, 66.0/255, 138.0/255),
            'hmof': (255.0/255, 153.0/255, 51.0/255),
            'qmof': (47.0/255, 108.0/255, 157.0/255),
            'tobacco': (63.0/255, 143.0/255, 61.0/255)
            }

    linkages = ['(Acyl)hydrazone', '(Keto)enamine', 'Azine', 'Imine', 'Boronate Ester', 'Oxazoline', 'Imide', 'Borosilicate', 'Borazine', 'Boroxine', 'Triazine']

    labels = {
            'di': r'Largest included diameter [$\mathregular{nm}$]',
            'rho': r'Mass density [$\mathregular{kg/m^3}$]',
            'asa_m': r'Gravimetric accessible surface area [$\mathregular{m^2/g}$]',
            'asa_v': r'Volumetric accessible surface area [$\mathregular{m^2/cm^3}$]',
            'av_m': r'Gravimetric accessible volume [$\mathregular{cm^3/g}$]',
            'av_v': r'Pore fraction [-]',
            }
    units = {
            'di': nanometer,
            'rho': kilogram/(meter**3),
            'asa_m': meter**2/gram,
            'asa_v': meter**2/centimeter**3,
            'av_m': centimeter**3/gram,
            'av_v': 1
            }
    
    lims = {
            'di': (0.0, 0.0),
            'rho': (0.0, 2000.0),
            'asa_m': (0.0, 11311.16862808182),
            'asa_v': (0.0, 3144.424778819206),
            'av_m': (0.0, 0.0),
            'av_v': (0.0, 1.0)
            }

    data = get_data(databases = ['redd-coffee', 'hmof', 'qmof', 'tobacco', 'iza'])
    
    for xfeat, yfeat in [('asa_m', 'rho'), ('asa_m', 'asa_v'), ('av_v', 'asa_v'), ('av_v', 'asa_m'), ('av_v', 'di'), ('av_v', 'rho'), ('di', 'asa_v'), ('rho', 'asa_v')]:
        continue
        fig, ax = plt.subplots()
        cmaps = []
        for database in ['tobacco', 'hmof', 'qmof', 'iza', 'redd-coffee']:
            db_data = extract(data, database)
            db_data = db_data[db_data[xfeat] > 0.0]
            db_data = db_data[db_data[yfeat] > 0.0]
            x = np.array(db_data[xfeat].values)/units[xfeat]
            y = np.array(db_data[yfeat].values)/units[yfeat]
            cmaps.append(plot_xy_kernel(ax, x, y, create_colormap(database_colors[database]), N = 500, rasterized = True))

        ax.axis('auto')
        ax.set_xlabel(labels[xfeat])
        ax.set_ylabel(labels[yfeat])
        xlim = lims[xfeat]
        ylim = lims[yfeat]
        if not xlim[0] == xlim[1]:
            ax.set_xlim(xlim)
        if not ylim[0] == ylim[1]:
            ax.set_ylim(ylim)
        fig.savefig('../figs/mat_{}_{}.pdf'.format(xfeat, yfeat), bbox_inches = 'tight')

    for xfeat, yfeat in [('asa_m', 'asa_v'), ('asa_m', 'rho'), ('av_v', 'asa_v'), ('rho', 'di'), ('asa_m', 'di'), ('av_m', 'di'), ('av_v', 'di'), ('av_v', 'rho'), ('av_v', 'asa_m'), ('di', 'asa_v'), ('rho', 'asa_v'), ('asa_m', 'av_m')]:
        fig, ax = plt.subplots()
        cmaps = []
        for dim, cmap in zip([3, 2], ['Greens', 'YlOrRd']):
            db_data = extract(data, 'redd-coffee', dim = dim)
            db_data = db_data[db_data[xfeat] > 0.0]
            db_data = db_data[db_data[yfeat] > 0.0]
            x = np.array(db_data[xfeat].values)/units[xfeat]
            y = np.array(db_data[yfeat].values)/units[yfeat]
            cmap = colors.ListedColormap(cm.get_cmap(cmap)(np.linspace(0.25, 1.0, 192)))
            cmaps.append(ax.hexbin(x, y, cmap = cmap, norm = colors.LogNorm(), mincnt = 1, vmax = None, rasterized = True))
        ax.set_xlabel(labels[xfeat])
        ax.set_ylabel(labels[yfeat])
        xlim = lims[xfeat]
        ylim = lims[yfeat]
        if not xlim[0] == xlim[1]:
            ax.set_xlim(xlim)
        if not ylim[0] == ylim[1]:
            ax.set_ylim(ylim)
        fig.savefig('../figs/own_{}_{}.pdf'.format(xfeat, yfeat), bbox_inches = 'tight')
    
    for xfeat in ['di', 'rho', 'asa_v']:
        fig, ax = plt.subplots()
        p = cm.get_cmap('RdYlGn_r')
        x = np.linspace(0.0, 1.0, 256)
        L = 0.7
        vals = p(np.piecewise(x, [x < 0.5, x >= 0.5], [lambda x: L*x, lambda x: L*(x-1.0)+1.0]))
        p = colors.ListedColormap(vals)
        for i, linkage in enumerate(linkages):
            db_data = extract(data, 'redd-coffee', linkage = linkage)
            db_data = db_data[db_data[xfeat] > 0.0]
            x = np.array(db_data[xfeat].values)/units[xfeat]
            plot_x_kernel(ax, x, p(float(i)/(len(linkages) - 1)), label = linkage)
        ax.legend()
        ax.set_xlabel(labels[xfeat])
        ax.set_ylabel('Normalized frequency')
        ax.set_yticks([])
        if xfeat == 'rho':
            ax.set_xlim(0, 500)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, xmax)
        fig.savefig('../figs/hist_{}.pdf'.format(xfeat), bbox_inches = 'tight')



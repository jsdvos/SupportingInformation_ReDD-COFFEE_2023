import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from collections import Counter

from molmod.units import nanometer, centimeter, meter, gram, kilogram

def get_data(sort = False, sortkey = 'DELTA abs_ads [mg/g]'):
    # ADSORPTION for each structure
    # abs_ads [mg/g], abs_ads [cm3STP/cm3], q_st [kjmol]
    # for low pressure (LP = 5.8bar) and high pressure (HP = 65bar) and difference (DELTA = HP - LP)
    low_press = pd.read_csv('../../data/CH4_298K_580000Pa.csv', sep = ';')
    low_press.columns = ['LP ' + column if not column == 'struct' else column for column in low_press.columns]
    high_press = pd.read_csv('../../data/CH4_298K_6500000Pa.csv', sep = ';')
    high_press.columns = ['HP ' + column if not column == 'struct' else column for column in high_press.columns]
    adsorption = pd.merge(low_press, high_press, how = 'outer', on='struct')
    for column in adsorption.columns:
        if not column[-6:] in ['struct', '[mg/g]', 'P/cm3]', 'kjmol]']:
            adsorption.drop(column, axis = 1, inplace = True)
    for column in ['abs_ads [cm3STP/cm3]', 'abs_ads [mg/g]', 'q_st [kjmol]']:
        adsorption['DELTA ' + column] = adsorption['HP ' + column] - adsorption['LP ' + column]
    
    # TEXTURAL for each structure
    # rho, di, asa_m, av_m, asa_v, av_v
    textural = pd.read_csv('../../data/features_subset.csv', sep = ';', usecols = ['struct', 'database', 'vol', 'rho', 'di', 'df', 'dif', 'asa', 'av', 'nasa', 'nav'])
    textural['mass'] = textural['rho']*textural['vol']
    for feature in ['asa', 'av', 'nasa', 'nav']:
        textural[feature + '_m'] = textural[feature]/textural['mass']
        textural[feature + '_v'] = textural[feature]/textural['vol']
    textural.drop('vol', axis = 1, inplace = True)
    textural.drop('mass', axis = 1, inplace = True)
    textural.drop('df', axis = 1, inplace = True)
    textural.drop('dif', axis = 1, inplace = True)
    textural.drop('asa', axis = 1, inplace = True)
    textural.drop('av', axis = 1, inplace = True)
    textural.drop('nasa', axis = 1, inplace = True)
    textural.drop('nav', axis = 1, inplace = True)
    textural.drop('nasa_m', axis = 1, inplace = True)
    textural.drop('nav_m', axis = 1, inplace = True)
    textural.drop('nasa_v', axis = 1, inplace = True)
    textural.drop('nav_v', axis = 1, inplace = True)
    
    data = pd.merge(textural, adsorption, how = 'outer', on = 'struct')
    if sort:
        data.sort_values(by = [sortkey], ascending = False, inplace = True)
    return data

if __name__ == '__main__':
    sortkey = 'DELTA abs_ads [cm3STP/cm3]'
    perc = 0.05
    data = get_data(sort = True, sortkey = sortkey)

    project = 'main'
    #project = 'suppinfo'

    if project == 'main':
        # Figures main manuscript
        all_keys = ['DELTA abs_ads [mg/g]', 'DELTA abs_ads [cm3STP/cm3]', 'DELTA q_st [kjmol]']
        labels = {
                'di': 'Largest pore diameter [nm]',
                'av_v': 'Pore fraction [-]',
                'DELTA abs_ads [mg/g]': 'Gravimetric deliverable capacity [g/g]',
                'DELTA abs_ads [cm3STP/cm3]': 'Volumetric deliverable capacity [vSTP/v]',
                'DELTA q_st [kjmol]': 'Difference in heat of adsorption [kJ/mol]'
        }
        units = {
                'di': nanometer,
                'av_v': 1.0,
                'DELTA abs_ads [mg/g]': 1000.0,
                'DELTA abs_ads [cm3STP/cm3]': 1.0,
                'DELTA q_st [kjmol]': 1.0
        }
        data = data.dropna(subset = all_keys)

        cmap = colors.ListedColormap(cm.get_cmap('Greens')(np.linspace(0.25, 1.0, 192)))

        # Figure a: CH4_M vs CH4_V
        fig, ax = plt.subplots()
        xfeat = 'DELTA abs_ads [mg/g]'
        yfeat = 'DELTA abs_ads [cm3STP/cm3]'
        x = np.array(data[xfeat].values)/units[xfeat]
        y = np.array(data[yfeat].values)/units[yfeat]
        colormap = ax.hexbin(x, y, cmap = cmap, norm = colors.LogNorm(), mincnt = 1, vmax = 218.0, rasterized =True)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.axvline(0.5, ymin, ymax, ls = '--', c = 'k', lw = 1)
        ax.axhline(315, xmin, xmax, ls = '--', c = 'k', lw = 1)
        ax.set_xlabel(labels[xfeat])
        ax.set_ylabel(labels[yfeat])
        ax.text(xmax-0.02*(xmax-xmin), 315-0.02*(ymax-ymin), 'ARPA-E target\n315vSTP/v', ha = 'right', va = 'top', ma = 'center')
        ax.text(0.5+0.02*(xmax-xmin), 315-0.02*(ymax-ymin), 'ARPA-E target\n0.5g/g', ha = 'left', va = 'top', ma = 'center', rotation = 90)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        fig.savefig('../../figs/CH4M_CH4V.pdf', bbox_inches = 'tight')
        cmap_vals = colormap.get_array()
        assert max(cmap_vals) <= 218.0

        # Figure b: delta Qst vs delta CH4_V
        fig, ax = plt.subplots()
        xfeat = 'DELTA q_st [kjmol]'
        yfeat = 'DELTA abs_ads [cm3STP/cm3]'
        x = np.array(data[xfeat].values)/units[xfeat]
        y = np.array(data[yfeat].values)/units[yfeat]
        colormap = ax.hexbin(x, y, cmap = cmap, norm = colors.LogNorm(), mincnt = 1, vmax = 218.0, rasterized =True)
        ax.set_xlabel(labels[xfeat])
        ax.set_ylabel(labels[yfeat])
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)
        fig.savefig('../../figs/DQst_DCH4V.pdf', bbox_inches = 'tight')
        cmap_vals = colormap.get_array()
        assert max(cmap_vals) <= 218.0

        # Figure c: di vs CH4_V
        # Figure d: av_v vs CH4_V
        for xfeat, yfeat in [('di', 'DELTA abs_ads [cm3STP/cm3]'), ('av_v', 'DELTA abs_ads [cm3STP/cm3]')]:
            # Define boxes
            left = 0.1
            width = 0.8
            bottom = 0.1
            height = 0.65
            rect_hexbin = [left, bottom, width, height]
            rect_hist = [left, bottom + height, width, 0.2]

            plt.figure()
            x = np.array(data[xfeat].values)/units[xfeat]
            y = np.array(data[yfeat].values)/units[yfeat]
            cutoff_top = (y[int(perc*len(y)) - 1] + y[int(perc*len(y))])/2
            cutoff_bottom = (y[-int(perc*len(y)) - 1] + y[-int(perc*len(y))])/2
            # 2D hexagonal histogram
            ax_hexbin = plt.axes(rect_hexbin)
            ax_hexbin.tick_params(top = True)
            colormap = ax_hexbin.hexbin(x, y, cmap = cmap, norm = colors.LogNorm(), mincnt = 1, vmax = 218.0, rasterized =True)
            ax_hexbin.set_xlabel(labels[xfeat])
            ax_hexbin.set_ylabel(labels[yfeat])
            xmin, xmax = ax_hexbin.get_xlim()
            ymin, ymax = ax_hexbin.get_ylim()
            if xfeat == 'av_v':
                xmax = 1.0
            ax_hexbin.set_xlim(0, xmax)
            ax_hexbin.set_ylim(0, ymax)
            # 5% subset lines
            red = (183.0/255, 21.0/255, 22.0/255)
            green = (63.0/255, 143.0/255, 61.0/255)
            ax_hexbin.axhline(cutoff_top, ls = '--', c = green, lw = 1)
            ax_hexbin.axhline(cutoff_bottom, ls = '--', c = red, lw = 1)
            xmin, xmax = ax_hexbin.get_xlim()
            ymin, ymax = ax_hexbin.get_ylim()
            ax_hexbin.text(xmax - 0.01*(xmax-xmin), cutoff_top + 0.01*(ymax-ymin), 'TOP 5%', ha = 'right', va = 'bottom', c = green, weight = 'bold')
            ax_hexbin.text(xmax - 0.01*(xmax-xmin), cutoff_bottom - 0.01*(ymax-ymin), 'WORST 5%', ha = 'right', va = 'top', c = red, weight = 'bold')
            # Histogram
            ax_hist = plt.axes(rect_hist)
            x_top = x[:int(perc*len(y))]
            x_bottom = x[-int(perc*len(y)):]
            bins = np.linspace(min(x), max(x), 100)
            ax_hist.hist(x_bottom, alpha = 0.5, bins = bins, color = red, zorder = 1)
            ax_hist.hist(x_top, alpha = 0.5, bins = bins, color = green, zorder = 2)
            ax_hist.set_xlim(xmin, xmax)
            ymin, ymax = ax_hist.get_ylim()
            ax_hist.set_ylim(0, ymax)
            ax_hist.axis('off')
            plt.savefig('../../figs/{}_CH4.pdf'.format(xfeat.split()[0], yfeat), bbox_inches = 'tight')
            cmap_vals = colormap.get_array()
            assert max(cmap_vals) <= 218.0
            if xfeat == 'av_v':
                assert max(cmap_vals) == 218.0

    if project == 'suppinfo':
        # SI figures
        all_keys = ['DELTA abs_ads [mg/g]', 'DELTA abs_ads [cm3STP/cm3]', 'DELTA q_st [kjmol]']
        labels = {
                'di': r'Largest pore diameter [$\mathregular{nm}$]',
                'rho': r'Mass density [$\mathregular{kg/m^3}$]',
                'asa_m': r'Gravimetric accessible surface area [$\mathregular{m^2/g}$]',
                'asa_v': r'Volumetric accessible surface area [$\mathregular{m^2/cm^3}$]',
                'av_m': r'Gravimetric accessible volume [$\mathregular{cm^3/g}$]',
                'av_v': r'Pore fraction [$\mathregular{-}$]',
                'LP abs_ads [mg/g]': r'Gravimetric uptake at low pressure [$\mathregular{g/g}$]',
                'LP abs_ads [cm3STP/cm3]': r'Volumetric uptake at low pressure [$\mathregular{vSTP/v]}$',
                'LP q_st [kjmol]': r'Heat of adsorption at low pressure [$\mathregular{kJ/mol}$]',
                'HP abs_ads [mg/g]': r'Gravimetric uptake at high pressure [$\mathregular{g/g}$]',
                'HP abs_ads [cm3STP/cm3]': r'Volumetric uptake at high pressure [$\mathregular{vSTP/v}$]',
                'HP q_st [kjmol]': r'Heat of adsorption at high pressure [$\mathregular{kJ/mol}$]',
                'DELTA abs_ads [mg/g]': r'Gravimetric deliverable capacity [$\mathregular{g/g}$]',
                'DELTA abs_ads [cm3STP/cm3]': r'Volumetric deliverable capacity [$\mathregular{vSTP/v}$]',
                'DELTA q_st [kjmol]': r'Difference in heat of adsorption [$\mathregular{kJ/mol}$]'
        }
        units = {
                'di': nanometer,
                'rho': kilogram/(meter**3),
                'asa_m': meter**2/gram,
                'asa_v': meter**2/(centimeter**3),
                'av_m': centimeter**3/gram,
                'av_v': 1.0,
                'LP abs_ads [mg/g]': 1000.0,
                'LP abs_ads [cm3STP/cm3]': 1.0,
                'LP q_st [kjmol]': 1.0,
                'HP abs_ads [mg/g]': 1000.0,
                'HP abs_ads [cm3STP/cm3]': 1.0,
                'HP q_st [kjmol]': 1.0,
                'DELTA abs_ads [mg/g]': 1000.0,
                'DELTA abs_ads [cm3STP/cm3]': 1.0,
                'DELTA q_st [kjmol]': 1.0
        }
        fig_names = {
                'di': 'di',
                'rho': 'rho',
                'asa_m': 'asa_m',
                'asa_v': 'asa_v',
                'av_m': 'av_m',
                'av_v': 'av_v',
                'LP abs_ads [mg/g]': 'LP_M',
                'LP abs_ads [cm3STP/cm3]': 'LP_V',
                'LP q_st [kjmol]': 'LP_Q',
                'HP abs_ads [mg/g]': 'HP_M',
                'HP abs_ads [cm3STP/cm3]': 'HP_V',
                'HP q_st [kjmol]': 'HP_Q',
                'DELTA abs_ads [mg/g]': 'DELTA_M',
                'DELTA abs_ads [cm3STP/cm3]': 'DELTA_V',
                'DELTA q_st [kjmol]': 'DELTA_Q'

                }
        data = data.dropna(subset = all_keys)

        cmap = colors.ListedColormap(cm.get_cmap('Greens')(np.linspace(0.25, 1.0, 192)))
        cmap_ylorrd = colors.ListedColormap(cm.get_cmap('YlOrRd')(np.linspace(0.25, 1.0, 192))) 
        for xfeat, yfeat in [('av_m', 'LP abs_ads [mg/g]'), ('av_m', 'HP abs_ads [mg/g]'), ('av_m', 'DELTA abs_ads [mg/g]'), ('asa_v', 'LP abs_ads [cm3STP/cm3]'), ('asa_v', 'HP abs_ads [cm3STP/cm3]'), ('asa_v', 'DELTA abs_ads [cm3STP/cm3]'), ('di', 'LP q_st [kjmol]'), ('asa_m', 'LP q_st [kjmol]'), ('av_v', 'LP q_st [kjmol]'), ('di', 'HP q_st [kjmol]'), ('asa_m', 'HP q_st [kjmol]'), ('av_v', 'HP q_st [kjmol]')]:
            fig, ax = plt.subplots()
            x = np.array(data[xfeat].values)/units[xfeat]
            y = np.array(data[yfeat].values)/units[yfeat]
            colormap = ax.hexbin(x, y, cmap = cmap, norm = colors.LogNorm(), mincnt = 1, rasterized =True)
            ax.set_xlabel(labels[xfeat])
            ax.set_ylabel(labels[yfeat])
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax)
            xfeat = fig_names[xfeat]
            yfeat = fig_names[yfeat]
            fig.savefig('../../figs/{}_{}.pdf'.format(xfeat, yfeat), bbox_inches = 'tight', dpi = 200)

        tops_2d = os.listdir('../../../../DatabaseGeneration/StructureAssembly/data/Topologies/2D')
        tops_3d = os.listdir('../../../../DatabaseGeneration/StructureAssembly/data/Topologies/3D')
        for xfeat, yfeat in [('DELTA abs_ads [mg/g]', 'DELTA abs_ads [cm3STP/cm3]'), ('DELTA q_st [kjmol]', 'DELTA abs_ads [cm3STP/cm3]'), ('di', 'DELTA abs_ads [cm3STP/cm3]'), ('av_v', 'DELTA abs_ads [cm3STP/cm3]')]:
            fig, ax = plt.subplots()
            x = np.array(data[xfeat].values)/units[xfeat]
            y = np.array(data[yfeat].values)/units[yfeat]
            structs = data['struct'].values
            x_2d = []
            y_2d = []
            x_3d = []
            y_3d = []
            for i in range(len(structs)):
                struct = structs[i]
                top = struct.split('_')[0]
                if top + '.top' in tops_2d:
                    x_2d.append(x[i])
                    y_2d.append(y[i])
                elif top + '.top' in tops_3d:
                    x_3d.append(x[i])
                    y_3d.append(y[i])
                else:
                    raise RuntimeError('Could not find topology {}'.format(top))
            colormap_3d = ax.hexbin(x_3d, y_3d, cmap = cmap, norm = colors.LogNorm(), mincnt = 1, rasterized =True)
            colormap_2d = ax.hexbin(x_2d, y_2d, cmap = cmap_ylorrd, norm = colors.LogNorm(), mincnt = 1, rasterized =True)
            ax.set_xlabel(labels[xfeat])
            ax.set_ylabel(labels[yfeat])
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax)
            xfeat = fig_names[xfeat]
            yfeat = fig_names[yfeat]
            fig.savefig('../../figs/{}_{}_XD.pdf'.format(xfeat, yfeat), bbox_inches = 'tight', dpi = 200)



        for xfeat, yfeat in [('rho', 'DELTA abs_ads [cm3STP/cm3]'), ('asa_m', 'DELTA abs_ads [cm3STP/cm3]'), ('di', 'DELTA abs_ads [mg/g]'), ('rho', 'DELTA abs_ads [mg/g]'), ('asa_m', 'DELTA abs_ads [mg/g]'), ('av_v', 'DELTA abs_ads [mg/g]')]:
            # Define boxes
            left = 0.1
            width = 0.8
            bottom = 0.1
            height = 0.65
            #rect_hexbin = [left, bottom, width, height]
            #rect_hist = [left, bottom + height, width, 0.2]
            rect_hexbin = [left, bottom, width, height]
            rect_hist = [left, bottom + height, width, 0.2]
            rect_colorbar = [left + width + 0.05, bottom, 0.03, height]

            plt.figure()
            x = np.array(data[xfeat].values)/units[xfeat]
            y = np.array(data[yfeat].values)/units[yfeat]
            s = sorted(zip(y, x), reverse = True)
            x = np.array([b for a, b in s])
            y = np.array([a for a, b in s])
            cutoff_top = (y[int(perc*len(y)) - 1] + y[int(perc*len(y))])/2
            cutoff_bottom = (y[-int(perc*len(y)) - 1] + y[-int(perc*len(y))])/2
            # 2D hexagonal histogram
            ax_hexbin = plt.axes(rect_hexbin)
            ax_hexbin.tick_params(top = True)
            colormap = ax_hexbin.hexbin(x, y, cmap = cmap, norm = colors.LogNorm(), mincnt = 1, rasterized =True)
            ax_hexbin.set_xlabel(labels[xfeat])
            ax_hexbin.set_ylabel(labels[yfeat])
            xmin, xmax = ax_hexbin.get_xlim()
            ymin, ymax = ax_hexbin.get_ylim()
            if xfeat == 'av_v':
                xmax = 1.0
            ax_hexbin.set_xlim(0, xmax)
            ax_hexbin.set_ylim(0, ymax)
            # 5% subset lines
            red = (183.0/255, 21.0/255, 22.0/255)
            green = (63.0/255, 143.0/255, 61.0/255)
            ax_hexbin.axhline(cutoff_top, ls = '--', c = green, lw = 1)
            ax_hexbin.axhline(cutoff_bottom, ls = '--', c = red, lw = 1)
            xmin, xmax = ax_hexbin.get_xlim()
            ymin, ymax = ax_hexbin.get_ylim()

            xtop = xmax - 0.01*(xmax-xmin)
            ytop = cutoff_top + 0.01*(ymax-ymin)
            hatop = 'right'
            vatop = 'bottom'
            xbot = xmax - 0.01*(xmax-xmin)
            ybot = cutoff_bottom - 0.01*(ymax-ymin)
            habot = 'right'
            vabot = 'top'
            if (xfeat, yfeat) == ('asa_m', 'DELTA abs_ads [mg/g]'):
                ybot = cutoff_bottom + 0.001*(ymax - ymin)
                vabot = 'bottom'
                xbot = xmax - 0.005*(xmax-xmin)
                xtop = xmax - 0.005*(xmax-xmin)
            elif (xfeat, yfeat) == ('asa_m', 'DELTA abs_ads [cm3STP/cm3]'):
                xbot = 3000
                habot = 'left'
            elif (xfeat, yfeat) == ('rho', 'DELTA abs_ads [cm3STP/cm3]'):
                pass
            elif (xfeat, yfeat) == ('rho', 'DELTA abs_ads [mg/g]'):
                ybot = cutoff_bottom + 0.01*(ymax - ymin)
                vabot = 'bottom'
            elif (xfeat, yfeat) == ('di', 'DELTA abs_ads [mg/g]'):
                xtop = xmax - 0.005*(xmax-xmin)
                ytop = cutoff_top - 0.02*(ymax-ymin)
                vatop = 'top'
                xbot = xmax - 0.005*(xmax-xmin)
                ybot = cutoff_bottom + 0.02*(ymax - ymin)
                vabot = 'bottom'
            elif (xfeat, yfeat) == ('av_v', 'DELTA abs_ads [mg/g]'):
                xtop = 0.01*(xmax-xmin)
                xbot = 0.01*(xmax - xmin)
                ybot = cutoff_bottom + 0.02*(ymax - ymin)
                ytop = cutoff_top + 0.02*(ymax - ymin)
                vabot = 'bottom'
                habot = 'left'
                hatop = 'left'
            ax_hexbin.text(xtop, ytop, 'TOP 5%', ha = hatop, va = vatop, c = green, weight = 'bold')
            ax_hexbin.text(xbot, ybot, 'WORST 5%', ha = habot, va = vabot, c = red, weight = 'bold')
            # Histogram
            ax_hist = plt.axes(rect_hist)
            x_top = x[:int(perc*len(y))]
            x_bottom = x[-int(perc*len(y)):]
            bins = np.linspace(min(x), max(x), 100)
            ax_hist.hist(x_bottom, alpha = 0.5, bins = bins, color = red, zorder = 1)
            ax_hist.hist(x_top, alpha = 0.5, bins = bins, color = green, zorder = 2)
            ax_hist.set_xlim(xmin, xmax)
            ymin, ymax = ax_hist.get_ylim()
            ax_hist.set_ylim(0, ymax)
            ax_hist.axis('off')
            xfeat = fig_names[xfeat]
            yfeat = fig_names[yfeat]
            plt.savefig('../../figs/{}_{}_hist.pdf'.format(xfeat, yfeat), bbox_inches = 'tight')



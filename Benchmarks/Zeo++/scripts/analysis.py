import os

import numpy as np
import matplotlib.pyplot as plt

from molmod.units import angstrom, centimeter, meter, gram, kilogram, nanometer

def read_res(fn):
    with open(fn, 'r') as f:
        words = f.readlines()[0].strip().split()
        assert words[0] == os.path.basename(fn)
        di = float(words[1])*angstrom
        df = float(words[2])*angstrom
        dif = float(words[3])*angstrom
    return di, df, dif

def read_sa(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
        words = lines[0].strip().split()
        assert words[0] == '@'
        assert words[1] == os.path.basename(fn)
        assert words[2] == 'Unitcell_volume:'
        volume = float(words[3])*angstrom**3
        assert words[4] == 'Density:'
        rho = float(words[5])*gram/centimeter**3
        assert words[6] == 'ASA_A^2:'
        asa = float(words[7])*angstrom**2
        assert words[8] == 'ASA_m^2/cm^3:'
        assert abs(float(words[9])*(meter**2/centimeter**3) - asa/volume) < 1e-3
        assert words[10] == 'ASA_m^2/g:'
        assert abs(float(words[11])*meter**2/gram - asa/(volume*rho)) < 1e-3
        assert words[12] == 'NASA_A^2:'
        nasa = float(words[13])*angstrom**2
        assert words[14] == 'NASA_m^2/cm^3:'
        assert abs(float(words[15])*(meter**2/centimeter**3) - nasa/volume) < 1e-3
        assert words[16] == 'NASA_m^2/g:'
        assert abs(float(words[17])*(meter**2/gram) - nasa/(volume*rho)) < 1e-3
    return volume, rho, asa, nasa

def read_vol(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
        words = lines[0].strip().split()
        assert words[0] == '@'
        assert words[1] == os.path.basename(fn)
        assert words[2] == 'Unitcell_volume:'
        volume = float(words[3])*angstrom**3
        assert words[4] == 'Density:'
        rho = float(words[5])*gram/centimeter**3
        assert words[6] == 'AV_A^3:'
        av = float(words[7])*angstrom**3
        assert words[8] == 'AV_Volume_fraction:'
        assert abs(float(words[9]) - av/volume)  < 1e-3
        assert words[10] == 'AV_cm^3/g:'
        assert (float(words[11])*centimeter**3/gram - av/(volume*rho)) < 1e-3
        assert words[12] == 'NAV_A^3:'
        nav = float(words[13])*angstrom**3
        assert words[14] == 'NAV_Volume_fraction:'
        assert abs(float(words[15]) - nav/volume)  < 1e-3
        assert words[16] == 'NAV_cm^3/g:'
        assert (float(words[17])*centimeter**3/gram - nav/(volume*rho)) < 1e-3
    return volume, rho, av, nav

def read(struct, N):
    src_path = os.path.join('../data', struct)
    di, df, dif = read_res(os.path.join(src_path, '{}_optimized_{}.res'.format(struct, N)))
    volume_sa, rho_sa, asa, nasa = read_sa(os.path.join(src_path, '{}_optimized_{}.sa'.format(struct, N)))
    volume_vol, rho_vol, av, nav = read_vol(os.path.join(src_path, '{}_optimized_{}.vol'.format(struct, N)))
    assert abs(volume_sa - volume_vol) < 1e-3
    assert abs(rho_sa - rho_vol) < 1e-3
    return volume_vol, rho_vol, di, df, dif, asa, av, nasa, nav

if __name__ == '__main__':
    result = {key: [] for key in ['N', 'V', 'rho', 'di', 'df', 'dif', 'asa', 'av', 'nasa', 'nav']}
    unit_labels = {
            'N': r'Monte Carlo samples',
            'V': r'Volume [$\mathregular{nm^3}$]',
            'rho': r'Mass density [$\mathregular{kg/m^3}$]',
            'di': r'Largest included diameter [$\mathregular{nm}$]',
            'df': r'Largest diameter of a free sphere [$\mathregular{nm}$]',
            'dif': r'Largest included diameter along path of a free sphere [$\mathregular{nm}$]',
            'asa': r'Accessible surface area [$\mathregular{nm^2}$]',
            'av': r'Accessible volume [$\mathregular{nm^3}$]',
            'nasa': r'Non-accessible surface area [$\mathregular{nm^2}$]',
            'nav': r'Non-accessible volume [$\mathregular{nm^3}$]',
            }
    units = {
            'N': 1.0,
            'V': nanometer**3,
            'rho': kilogram/(meter**3),
            'di': nanometer,
            'df': nanometer,
            'dif': nanometer,
            'asa': nanometer**2,
            'av': nanometer**3,
            'nasa': nanometer**2,
            'nav': nanometer**3,
            }

    for struct in os.listdir('../data'):
        vs = []
        rhos = []
        dis = []
        dfs = []
        difs = []
        asas = []
        avs = []
        nasas = []
        navs = []
        Ns = []
        for N in [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]:
            v, rho, di, df, dif, asa, av, nasa, nav = read(struct, N)
            Ns.append(N)
            vs.append(v)
            rhos.append(rho)
            dis.append(di)
            dfs.append(df)
            difs.append(dif)
            asas.append(asa)
            avs.append(av)
            nasas.append(nasa)
            navs.append(nav)
        if not len(Ns) == len([250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]): continue
        result['N'].append(Ns)
        result['V'].append(vs)
        result['rho'].append(rhos)
        result['di'].append(dis)
        result['df'].append(dfs)
        result['dif'].append(difs)
        result['asa'].append(asas)
        result['av'].append(avs)
        result['nasa'].append(nasas)
        result['nav'].append(navs)
    
    from matplotlib import colors
    from matplotlib import cm
    p = cm.get_cmap('RdYlGn_r')
    x = np.linspace(0.0, 1.0, 256)
    L = 0.7
    vals = p(np.piecewise(x, [x < 0.5, x >= 0.5], [lambda x: L*x, lambda x: L*(x-1.0)+1.0]))
    p = colors.ListedColormap(vals)

    for key, value in result.items():
        if key not in ['asa', 'av']: continue
        Ns = result['N'][0]
        assert Ns == [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]
        cmax = max([data[-1] for data in value])
        if cmax == 0:
            print(key)
            for data in value:
                print(data)
            continue
        
        plt.figure()
        for data in value:
            plt.plot(Ns, np.array(data)/units[key], c = p(data[-1]/cmax))
        ymin, ymax = plt.ylim()
        plt.vlines(3000, ymin, ymax, 'k', '--')
        plt.ylim(ymin, ymax)
        plt.xlabel('Monte Carlo samples')
        plt.ylabel(unit_labels[key])
        plt.savefig(os.path.join('../figs', key + '.pdf'))

        plt.figure()
        for data in value:
            plt.plot(Ns, 100*(np.array(data) - data[-1])/data[-1], c = p(data[-1]/cmax))
        if key == 'asa':
            plt.ylim(-2, 2)
        elif key == 'av':
            plt.ylim(-5, 5)
        ymin, ymax = plt.ylim()
        plt.vlines(3000, ymin, ymax, 'k', '--')
        plt.ylim(ymin, ymax)
        plt.xlabel('Monte Carlo samples')
        plt.ylabel('Deviation of the {}[%]'.format(unit_labels[key].split('[')[0].lower()))
        plt.savefig(os.path.join('../figs', key + '_norm.pdf'))

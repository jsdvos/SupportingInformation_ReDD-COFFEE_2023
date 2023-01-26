import os

import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import colors

from yaff.system import System
from yaff.pes.generator import *
from yaff.pes.parameters import Parameters

def process_line(line):
    words = line[1].strip().split()
    key = []
    pars = []
    for word in words:
        try:
            pars.append(float(word))
        except ValueError:
            key.append(word)
    return tuple(key), tuple(pars)

def iter_matches(default, extended, generator):
    ffatypes_extended_to_default = {
            'C_B_BR_BDBA': 'C_HC2_01-01-02',
            'C_B_BDBA': 'C_BC2_01-01-02',
            'H_B_BR_BDBA': 'H_C_01-01-02',
            'C_O_BR_HHTP': 'C_HC2_18-08-01',
            'C_BR_HHTP': 'C_C3_18-08-01',
            'C_O_HHTP': 'C_C2O_18-08-01',
            'O_HHTP': 'O_BC_18-08-01',
            'B_HHTP': 'B_CO2_18-08-01',
            'H_O_BR_HHTP': 'H_C_18-08-01',
            'C_TP_PDBA': 'C_C3_H2C4_04-01-02',
            'C_TP_I_PDBA': 'C_C3_C6_04-01-02',
            'C_TP_O_PDBA': 'C_HC2_HC3_04-01-02',
            'C_B_BR_O_PDBA': 'C_HC2_BC3_04-01-02',
            'C_B_PDBA': 'C_BC2_H2C2O2_04-01-02',
            'H_B_BR_O_PDBA': 'H0_C_C2_04-01-02',
            'H_TP_O_PDBA': 'H1_C_C2_04-01-02',
            'B_THB': 'B_CO2_06-08-01',
            'O_THB': 'O_BC_06-08-01',
            'C_O_THB': 'C_C2O_06-08-01',
            'C_O_BR_THB': 'C_HC2_06-08-01',
            'H_O_BR_THB': 'H_C_06-08-01'
            }
    for line in default:
        key, pars = process_line(line)
        found = False
        alpha = float(sum([not ffa.endswith('_term') for ffa in key]))/len(key)
        for other_line in extended:
            other_key_old, other_pars = process_line(other_line)
            other_key = tuple([ffatypes_extended_to_default.setdefault(ffa_old, ffa_old) for ffa_old in other_key_old])
            if not generator.prefix == 'FIXQ':
                for equiv_key, equiv_pars in generator.iter_equiv_keys_and_pars(generator, key, pars):
                    if equiv_key == other_key:
                        found = True
                        yield equiv_pars, other_pars, alpha
            elif len(pars) == 1:
                for equiv_key, equiv_pars in [(key, pars), (key[::-1], (-pars[0],))]:
                    if equiv_key == other_key:
                        found = True
                        yield equiv_pars, other_pars, alpha
            elif len(pars) == 2:
                for equiv_key, equiv_pars in [(key, pars), (key[::-1], pars)]:
                    if equiv_key == other_key:
                        found = True
                        yield equiv_pars, other_pars, alpha


        if not found and not all([ffa.endswith('_term') for ffa in key]):
            print('Couldnt find a matching line for {}'.format(line))
        
sbus = ['01-01-02', '04-01-02', '06-08-01', '18-08-01']
pars_default = {sbu: Parameters.from_file([os.path.join('../../data', sbu, fn) for fn in ['pars_yaff.txt', 'pars_ei.txt']]) for sbu in sbus}
pars_extended = {sbu: Parameters.from_file([os.path.join('../../data', sbu + '_extended', fn) for fn in ['pars_yaff.txt', 'pars_ei.txt']]) for sbu in sbus}

for prefix in ['BONDHARM', 'BENDAHARM', 'TORSION', 'OOPDIST', 'CROSS', 'FIXQ']:
    generator = {'BONDHARM': BondHarmGenerator,
                'BENDAHARM': BendAngleHarmGenerator,
                'TORSION': TorsionGenerator,
                'OOPDIST': OopDistGenerator,
                'CROSS': CrossGenerator,
                'FIXQ': FixedChargeGenerator}[prefix]
    for count, par_info in enumerate(generator.par_info):
        par_name = par_info[0]
        if prefix == 'FIXQ' and par_name in ['R', 'Q0']:
            suffix = 'ATOM'
            count = {'Q0': 0, 'R': 1}[par_name]
        if prefix == 'FIXQ' and par_name == 'P':
            suffix = 'BOND'
            count = 0
        if not prefix == 'FIXQ': suffix = 'PARS'
        vals_default = []
        vals_extended = []
        alphas = []
        for sbu in sbus:
            lines_default = pars_default[sbu].sections[prefix].definitions[suffix].lines
            lines_extended = pars_extended[sbu].sections[prefix].definitions[suffix].lines
            for par_default, par_extended, alpha in iter_matches(lines_default, lines_extended, generator):
                if alpha > 0.0:
                    vals_default.append(par_default[count])
                    vals_extended.append(par_extended[count])
                    alphas.append(alpha)
        
        vals_default = np.array(vals_default)
        vals_extended = np.array(vals_extended)
        alphas = np.array(alphas)

        fig, ax = plt.subplots()
        label = {
                'BONDHARM': {
                    'K': r'$K$ [$\frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{\AA}^2}$]',
                    'R0': r'$R_0$ [$\mathrm{\AA}$]'
                    },
                'BENDAHARM': {
                    'K': r'$K$ [$\frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{rad}^2}$]',
                    'THETA0': r'$\Theta_0$ [$^\circ$]'
                    },
                'TORSION': {
                    'M': r'$M$ [-]',
                    'A': r'$A$ [$\frac{\mathrm{kJ}}{\mathrm{mol}}$]',
                    'PHI0': r'$\Phi_0$ [$^\circ$]'
                    },
                'OOPDIST': {
                    'K': r'$K$ [$\frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{\AA}^2}$]',
                    'D0': r'$D_0$ [$\mathrm{\AA}$]'
                    },
                'CROSS': {
                    'KSS': r'$K_{S_0S_1}$ [$\frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{\AA}^2}$]',
                    'KBS0': r'$K_{BS_0}$ [$\frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{\AA}\ \mathrm{rad}}$]',
                    'KBS1': r'$K_{BS_1}$ [$\frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{\AA}\ \mathrm{rad}}$]',
                    'R0': r'$R^0_0$ [$\mathrm{\AA}$]',
                    'R1': r'$R^1_0$ [$\mathrm{\AA}$]',
                    'THETA0': r'$\Theta_0$ [$^\circ$]'
                    },
                'FIXQ': {
                    'R': r'$R$ [$\mathrm{\AA}$]',
                    'Q0': r'$Q_0$ [e]',
                    'P': r'$P$ [e]'
                    }
                }[prefix][par_name]
        ax.set_xlabel('{}\nDefault termination'.format(label))
        ax.set_ylabel('{}\nExtended termination'.format(label))
        p = cm.get_cmap('RdYlGn')
        x = np.linspace(0.0, 1.0, 256)
        L = 0.7
        vals = p(np.piecewise(x, [x < 0.5, x >= 0.5], [lambda x: L*x, lambda x: L*(x-1.0)+1.0]))
        p = colors.ListedColormap(vals)
        scatter = ax.scatter(vals_default, vals_extended, c = alphas, cmap = p, vmin = 0.0, vmax = 1.0)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lims = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax.plot(lims, lims, 'k--', linewidth = 0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        handles, labels = scatter.legend_elements()
        frac_labels = []
        for label in labels:
            result = re.search('mathdefault{(.*)}', label)
            frac_labels.append({
                '0.25': '1/4',
                '0.3333': '1/3',
                '0.5': '1/2',
                '0.50': '1/2',
                '0.6667': '2/3',
                '0.75': '3/4',
                '1': '1',
                '1.0': '1',
                '1.00': '1',
                '1.000': '1',
                '1.0000': '1'
                }[result.group(1)])
        ax.legend(handles = handles, labels = frac_labels, title = 'Rescaling factors')
        plt.savefig('../../figs/validate_terminations/{}_{}.pdf'.format(prefix, par_name), bbox_inches = 'tight')




import os

from yaff import System, log
log.set_level(0)
from molmod.units import angstrom, meter, centimeter, gram

# STEP 1: iterate over structures
def iter_struct(database):
    db_path_chk = {
            'redd-coffee': '/path/to/redd-coffee',
            'martin': '/path/to/martin',
            'mercado': '/path/to/mercado',
            'core': '/path/to/core',
            'curated': '/path/to/curated'
            }
    db_path_geo = {
            'redd-coffee': '/path/to/redd-coffee',
            'martin': '/path/to/martin',
            'mercado': '/path/to/mercado',
            'core': '/path/to/core',
            'curated': '/path/to/curated'
            }
    db_path_rac = {
            'redd-coffee': '/path/to/redd-coffee',
            'martin': '/path/to/martin',
            'mercado': '/path/to/mercado',
            'core': '/path/to/core',
            'curated': '/path/to/curated'
            }
    for fn_chk in os.listdir(db_path_chk[database]):
        struct = fn_chk.split('.')[0]
        fn_geo = os.path.join(db_path_geo[database], struct + '.geo')
        fn_rac = os.path.join(db_path_rac[database], struct + '.rac')
        yield struct, fn_geo, fn_rac

# STEP 2: read geometric descriptors
def read_geo(struct, fn_geo):
    with open(fn_geo, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
        ## Check buildup of file - res
        #assert lines[0][0] == '{}_optimized.res'.format(struct)
        assert lines[0][0].endswith('.res')
        ## Check buildup of file - sa
        assert lines[1][0] == '@'
        #assert lines[1][1] == '{}_optimized.sa'.format(struct)
        assert lines[1][1].endswith('.sa')
        assert lines[1][2] == 'Unitcell_volume:'
        assert lines[1][4] == 'Density:'
        assert lines[1][6] == 'ASA_A^2:'
        assert lines[1][8] == 'ASA_m^2/cm^3:'
        assert lines[1][10] == 'ASA_m^2/g:'
        assert lines[1][12] == 'NASA_A^2:'
        assert lines[1][14] == 'NASA_m^2/cm^3:'
        assert lines[1][16] == 'NASA_m^2/g:'
        ## Check buildup of file - vol
        assert lines[4][0] == '@'
        #assert lines[4][1] == '{}_optimized.vol'.format(struct)
        assert lines[4][1].endswith('.vol')
        assert lines[4][2] == 'Unitcell_volume:'
        assert lines[4][4] == 'Density:'
        assert lines[4][6] == 'AV_A^3:'
        assert lines[4][8] == 'AV_Volume_fraction:'
        assert lines[4][10] == 'AV_cm^3/g:'
        assert lines[4][12] == 'NAV_A^3:'
        assert lines[4][14] == 'NAV_Volume_fraction:'
        assert lines[4][16] == 'NAV_cm^3/g:'

        # Extract values - res
        di = float(lines[0][1])*angstrom
        df = float(lines[0][2])*angstrom
        dif = float(lines[0][3])*angstrom
        # Extract values - sa
        volume_sa = float(lines[1][3])*angstrom**3
        rho_sa = float(lines[1][5])*gram/centimeter**3
        asa = float(lines[1][7])*angstrom**2
        nasa = float(lines[1][13])*angstrom**2
        # Extract values - vol
        volume_vol = float(lines[4][3])*angstrom**3
        rho_vol = float(lines[4][5])*gram/centimeter**3
        av = float(lines[4][7])*angstrom**3
        nav = float(lines[4][13])*angstrom**3
        assert abs(volume_sa - volume_vol) < 1e-3
        assert abs(rho_sa - rho_vol) < 1e-3
        volume = volume_sa
        rho = rho_sa

        # Check numerical values - sa
        assert abs(float(lines[1][9])*(meter**2/centimeter**3) - asa/volume) < 1e-3, '{} vs {}'.format(float(lines[4][9])*(meter**2/centimeter**3), asa/volume)
        assert abs(float(lines[1][11])*meter**2/gram - asa/(volume*rho)) < 1e-3
        assert abs(float(lines[1][15])*(meter**2/centimeter**3) - nasa/volume) < 1e-3
        assert abs(float(lines[1][17])*(meter**2/gram) - nasa/(volume*rho)) < 1e-3
        # Check numerical values - vol
        assert abs(float(lines[4][9]) - av/volume)  < 1e-3
        assert abs(float(lines[4][11])*centimeter**3/gram - av/(volume*rho)) < 1e-3
        assert abs(float(lines[4][15]) - nav/volume)  < 1e-3
        assert abs(float(lines[4][17])*centimeter**3/gram - nav/(volume*rho)) < 1e-3
    result = {
            'vol': volume,
            'rho': rho,
            'di': di,
            'df': df,
            'dif': dif,
            'asa': asa,
            'av': av,
            'nasa': nasa,
            'nav': nav
            }
    return result

# STEP 3: read RAC descriptors
def read_rac(struct, fn_rac):
    result = {}
    with open(fn_rac, 'r') as f:
        keys = None
        for line in f.readlines():
            words = line.strip().split()
            if words[0] in ['RAC', 'LigandRAC', 'FullLinkerRAC', 'FunctionalGroupRAC', 'LinkerConnectingRAC']:
                if words[0] == 'RAC':
                    keys = words[1:]
                else:
                    prefix = {'LigandRAC': 'lig', 'FullLinkerRAC': 'link', 'FunctionalGroupRAC': 'func', 'LinkerConnectingRAC': 'conn'}[words[0]]
                    for i, value in enumerate(words[1:]):
                        result['{}_{}'.format(prefix, keys[i])] = float(value)
    return result

# STEP 4: Write output
def write(result, fn_out):
    keys = ['vol', 'rho', 'di', 'df', 'dif', 'asa', 'av', 'nasa', 'nav']
    for prefix in ['lig', 'link', 'func', 'conn']:
        for key in ['prod_I_0', 'prod_I_1', 'prod_I_2', 'prod_I_3', 'prod_T_0', 'prod_T_1', 'prod_T_2', 'prod_T_3', 'prod_X_0', 'prod_X_1', 'prod_X_2', 'prod_X_3', 'prod_S_0', 'prod_S_1', 'prod_S_2', 'prod_S_3', 'prod_Z_0', 'prod_Z_1', 'prod_Z_2', 'prod_Z_3', 'prod_a_0', 'prod_a_1', 'prod_a_2', 'prod_a_3', 'diff_I_0', 'diff_I_1', 'diff_I_2', 'diff_I_3', 'diff_T_0', 'diff_T_1', 'diff_T_2', 'diff_T_3', 'diff_X_0', 'diff_X_1', 'diff_X_2', 'diff_X_3', 'diff_S_0', 'diff_S_1', 'diff_S_2', 'diff_S_3', 'diff_Z_0', 'diff_Z_1', 'diff_Z_2', 'diff_Z_3', 'diff_a_0', 'diff_a_1', 'diff_a_2', 'diff_a_3', 'N_start', 'N_part']:
            keys.append(prefix + '_' + key)
    with open(fn_out, 'w') as f:
        f.write('struct;database;' + ';'.join(keys) + '\n')
        sorted_struct = sorted(result.keys())
        for struct in sorted_struct:
            f.write('{};{};'.format(struct, result[struct]['database']) + ';'.join([str(result[struct].get(key)) for key in keys]) + '\n')

if __name__ == '__main__':
    result = {}
    for database in ['redd-coffee', 'martin', 'mercado', 'core', 'curated']:
        for struct, fn_geo, fn_rac in iter_struct(database = database):
            data = {'database': database}
            data.update(read_geo(struct, fn_geo))
            data.update(read_rac(struct, fn_rac))
            result[struct] = data
    write(result, '../../data/features.csv')



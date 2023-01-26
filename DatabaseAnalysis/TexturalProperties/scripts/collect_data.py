import os

from molmod.units import angstrom, meter, centimeter, gram

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
    return volume, rho, di, df, dif, asa, av, nasa, nav

def iter_struct(database):
    db_path = {
            'redd-coffee': '/path/to/redd-coffee'
            'hmof': '/path/to/hmof',
            'tobacco': '/path/to/tobacco',
            'qmof': '/path/to/qmof',
            'iza': '/path/to/iza'
            }
    for fn_geo in os.listdir(db_path[database]):
        struct = fn_geo.split('.')[0]
        yield struct, fn_geo


if __name__ == '__main__':
    with open('../data/structural.txt', 'w') as f:
        f.write('struct;database;vol;rho;di;df;dif;asa;av;nasa;nav\n')
        for database in ['redd-coffee', 'hmof', 'tobacco', 'qmof', 'iza']:
            for struct, fn_geo in iter_struct(database):
                vol, rho, di, df, dif, asa, av, nasa, nav = read_geo(fn_geo)
                f.write(';'.join([struct, database, vol, rho, di, df, dif, asa, av, nasa, nav]) + '\n')




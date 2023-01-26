import os
import numpy as np

def iter_struct(database):
    db_path_rac = {
            'ReDD-COFFEE': '/path/to/redd-coffee'
            'CoRE': '/path/to/core',
            'CURATED': '/path/to/curated',
            'Martin': '/path/to/martin',
            'Mercado': 'path/to/mercado'
            }[database]
    for fn_rac in os.listdir(db_path_rac):
        struct = fn_rac.split('.')[0]
        yield struct, database, fn_rac

def read_rac(fn):
    result = {}
    with open(fn, 'r') as f:
        for line in f:
            words = line.strip().split()
            if words[0] in ['RAC', 'LigandRAC', 'FullLinkerRAC', 'FunctionalGroupRAC', 'LinkerConnectingRAC']: continue
            if words[0].endswith('.chk'): continue
            ligand, count = words
            result[ligand] = count
    return result

def write(result, fn):
    ligands = sorted(['Borosilicate-tBu', 'Borosilicate-PropanoicAcid', 'BP-cube', 'Borosilicate-Me', 'Pyrimidazole4_LZU-564', 'Propargylamine', 'Aminal', 'Salen', 'Pyrimidazole1_LZU-561', 'Pyrimidazole2_LZU-562', 'Pyrimidazole3_LZU-563', 'Salen-Zn', 'PhenylQuinoline', 'Borosilicate', 'TetraHydroPyranQuinoline', '2106', '2015', 'Silicate-Li', 'Squaraine', 'Borazine', 'Ketoenamine', 'AzineH', 'HydrazoneH', 'Carbamate', 'ThioCarbamate', 'Silicate-Na', 'Hydrazone', 'C-C(CN)', 'Boroxine', 'Azine', 'Triazine', 'Phosphazene', 'BP-rod', 'Spiroborate-Li', 'Alpha-AminoNitrile', 'Imide', 'Imide6', 'Amine', 'Spiroborate', 'Olefin(CN)', 'Furonitrile', 'Benzimidazole', 'ImineTG+', 'ImineCo', 'Amide', 'Azodioxy', 'Enamine', 'Olefin', 'Olefin-CNterm', 'EnamineN', 'Boronate', 'Imine', 'Benzobisoxazole', 'Thiazole', 'ImineCH2', 'ImineNC', 'ImineUnphysical2', 'BOImine', 'Ester', 'ImineUnphysical', 'Dioxin', 'Phenazine', 'C-C', 'C-N'])
    keys = ['database'] + ligands
    with open(fn, 'w') as f:
        f.write(';'.join(['struct'] + keys))
        f.write('\n')
        for db in ['ReDD-COFFEE', 'CoRE', 'CURATED', 'Martin', 'Mercado']:
            for struct, db, fn_rac in iter_struct(database = db):
                values = [result[struct][key] for key in keys]
                f.write(';'.join([struct] + values))
                f.write('\n')

if __name__ == '__main__':
    result = {}
    for db in ['ReDD-COFFEE', 'CoRE', 'CURATED', 'Martin', 'Mercado']:
        for struct, db, fn_rac in iter_struct(db):
            data = {'database': db}
            data.update(read_rac(fn_rac))
            result[struct] = data
    write(result, '../../data/count_linkages/ligands.csv')




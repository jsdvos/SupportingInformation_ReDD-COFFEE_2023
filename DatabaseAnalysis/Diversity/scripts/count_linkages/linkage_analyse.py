import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linkage_filter import linkages

def process_data(df):
    for key, value in linkages.items():
        if '_' + value in df.columns:
            df['_' + value] += df[key]
        else:
            df['_' + value] = df[key]
    for column in df.columns:
        if column.startswith('_') or column in ['struct', 'database']:
            continue
        else:
            df.drop(column, inplace = True, axis = 1)
    df['_Mixed'] = df.astype(bool).sum(axis = 1) > 3
    own = df[['_' in db for db in df.database.values]]
    core = df[df.database == 'CoRE']
    curated = df[df.database == 'CURATED']
    martin = df[df.database == 'Martin']
    mercado = df[df.database == 'Mercado']
    return own, core, curated, martin, mercado

def count_occurences(db):
    count = db[db['_Mixed'] == 0].astype(bool).sum(axis = 0)
    count['_Mixed'] = db['_Mixed'].astype(bool).sum(axis = 0)
    return count

def plot_count(count, db, fn_fig):
    keys = ['Imine', 'Boronate Ester', 'Enamine', 'Triazine', 'Hydrazone', 'Azine', 'Imide', 'Boroxine', 'Borosilicate', 'Benzobisoxazole', 'Borazine', 'Amide', 'Amine', 'Carbon-Carbon', 'Olefin', 'Mixed', 'Other']
    assert len(keys) + 2 == len(count.keys()) # additional keys are struct and database
    counts = [float(count['_' + key])/len(db) for key in keys]
    y_pos = np.arange(len(keys))
    fig, ax = plt.subplots()
    ax.barh(y_pos, counts, align = 'center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    plt.xlim(0,1)
    plt.savefig(fn_fig, bbox_inches = 'tight')
    

if __name__ == '__main__':
    fn_data = '../../data/count_linkages/linkages.csv'
    df = pd.read_csv(fn_data, sep = ';')
    own, core, curated, martin, mercado = process_data(df)
    for db in [own, core, curated, martin, mercado]:
        db_name = db.iloc[0].database
        if db_name == '1_1':
            db_name = 'ReDD-COFFEE'
        count = count_occurences(db)
        tot = 0
        print(db_name)
        for key in ['Imine', 'Boronate Ester', 'Enamine', 'Triazine', 'Hydrazone', 'Azine', 'Imide', 'Boroxine', 'Borosilicate', 'Benzobisoxazole', 'Borazine', 'Amide', 'Amine', 'Carbon-Carbon', 'Olefin', 'Mixed', 'Other']:
            key = '_' + key
            print('{} -> {} ({:.2f}\\%)'.format(key[1:], count[key], 100*float(count[key])/len(db)))
        print('')
        plot_count(count, db, fn_fig = '../../figs/count_linkages/{}_linkages.pdf'.format(db_name))


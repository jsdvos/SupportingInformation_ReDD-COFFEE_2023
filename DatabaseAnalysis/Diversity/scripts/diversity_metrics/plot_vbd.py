import numpy as np
import matplotlib.pyplot as plt

# Get data
dbs = ['redd', 'core', 'curated', 'martin', 'mercado', 'subset']
fn_dat = '../../data/diversity_metrics/diversity_metrics.dat'
result = {}
with open(fn_dat, 'r') as f:
    keys = f.readline().strip().split()
    for line in f.readlines():
        values = line.strip().split()
        data = {keys[i]: float(values[i]) for i in range(1, len(keys))}
        result[values[0]] = data

# Plot
categories = ['V', 'B', 'D', 'V']
label_loc = np.linspace(0, 2*np.pi, len(categories))

for domain in result.keys():
    plt.figure(figsize=(8,8))
    plt.subplot(polar = True)
    for db in dbs:
        v = result[domain][db + '_v']
        b = result[domain][db + '_b']
        d = result[domain][db + '_d']
        if db == 'redd':
            db = 'redd-coffee'
        plt.plot(label_loc, [v, b, d, v], label = db)
    plt.ylim([0,1])
    plt.thetagrids(np.degrees(label_loc), labels = categories)
    plt.legend()
    plt.savefig('../../figs/diversity_metrics/{}.pdf'.format(domain))


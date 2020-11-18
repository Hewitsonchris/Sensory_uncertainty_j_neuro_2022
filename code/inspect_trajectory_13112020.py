import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datatable as dt
from scipy.interpolate import interp1d

fn = '../datta/G13.csv'
d = dt.fread(fn)
d.names = ('group', 'subject', 'trial', 'sig_mp', 'sig_ep', 'x', 'y')

d = d.to_pandas()

samp_rate = 0.01
ns = 100


def interp_traj(d):

    group = d['group'].to_numpy()[0]
    subject = d['subject'].to_numpy()[0]
    trial = d['trial'].to_numpy()[0]
    sig_mp = d['sig_mp'].to_numpy()[0]
    sig_ep = d['sig_ep'].to_numpy()[0]
    x = d['x'].to_numpy()
    y = d['y'].to_numpy()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    t = np.linspace(0, x.size * samp_rate, x.size)
    ts = np.linspace(t[0], t[-1], ns)
    t_ind = np.arange(0, ts.size, 1)
    fx = interp1d(t, x, kind='cubic')
    xs = fx(ts)
    fy = interp1d(t, y, kind='cubic')
    ys = fy(ts)
    d = pd.DataFrame({
        'group': group,
        'subject': subject,
        'trial': trial,
        'sig_mp': sig_mp,
        'sig_ep': sig_ep,
        'x': xs,
        'y': ys,
        't': ts,
        't_ind': t_ind
    })
    return d


ds = d.groupby(['group', 'subject', 'trial']).apply(interp_traj)
ds.reset_index(inplace=True, drop=True)

dds = ds.groupby(['sig_mp', 'sig_ep', 't_ind'])['x', 'y', 't'].mean()
dds = dds.reset_index()
dds = dds.sort_values(['sig_mp', 'sig_ep', 't_ind'])

# nrow = dds['sig_mp'].unique().size
# ncol = dds['sig_ep'].unique().size
# fig, ax = plt.subplots(nrow, ncol)
# for i, smp in enumerate(dds['sig_mp'].unique()):
#     for j, sep in enumerate(dds['sig_ep'].unique()):
#         dd = dds[(dds['sig_mp'] == smp) & (dds['sig_ep'] == sep)]
#         sns.scatterplot(data=dd, x='x', y='y', hue='sig_mp', ax=ax[i, j])

sns.scatterplot(data=dds,
                x='x',
                y='y',
                hue='sig_mp',
                style='sig_ep',
                palette='deep',
                s=100)
plt.show()

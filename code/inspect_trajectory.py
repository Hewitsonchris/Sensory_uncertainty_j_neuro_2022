import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datatable as dt
from scipy.interpolate import interp1d


def compute_traj(d):
    ds = d.groupby(['group', 'subject', 'trial', 'sig_mp',
                    'sig_ep']).apply(interp_traj)
    ds.reset_index(inplace=True, drop=True)

    dds = ds.groupby(['sig_mp', 'sig_ep',
                      't_ind'])['x', 'y', 't'].mean()

    dds = dds.reset_index()
    dds = dds.sort_values(['sig_mp', 'sig_ep', 't_ind'])

    return dds


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

    # plt.plot(x, y, '.')
    # plt.plot(xs, ys, '.')
    # plt.show()

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


def baseline_correct(d):
    sig_mp = d['sig_mp'].to_numpy()
    sig_ep = d['sig_ep'].to_numpy()
    x = d['x'].to_numpy()
    y = d['y'].to_numpy()
    t = d['t'].to_numpy()
    t_ind = d['t_ind'].to_numpy()

    x = x - d_base['x'].to_numpy()

    d = pd.DataFrame({
        'sig_mp': sig_mp,
        'sig_ep': sig_ep,
        'x': x,
        'y': y,
        't': t,
        't_ind': t_ind
    })

    return d


samp_rate = 0.01
ns = 100

fn = '../data/Base01314.csv'
d = dt.fread(fn)
d = d.to_pandas()
d['sig_ep'] = 4
d['sig_mp'] = 4
d_base = compute_traj(d)

fn = '../data/G345.csv'
d = dt.fread(fn)
d.names = ('group', 'subject', 'trial', 'sig_mp', 'sig_ep', 'x', 'y')
d = d.to_pandas()
d_adapt = compute_traj(d)

d_base = d_base.sort_values(['sig_ep', 'sig_mp', 't_ind'])
d_adapt = d_adapt.sort_values(['sig_ep', 'sig_mp', 't_ind'])

d_adapt_corrected = d_adapt.groupby(['sig_mp',
                                     'sig_ep']).apply(baseline_correct)
d_adapt_corrected.reset_index(drop=True, inplace=True)

s = 50
fig, ax = plt.subplots(1, 3)
sns.lineplot(
    data=d_base,
    x='x',
    y='y',
    hue='sig_mp',
    style='sig_ep',
    palette='deep',
    # s=s,
    ax=ax[0])
sns.lineplot(
    data=d_adapt,
    x='x',
    y='y',
    hue='sig_mp',
    style='sig_ep',
    palette='deep',
    # s=s,
    ax=ax[1])
sns.lineplot(
    data=d_adapt_corrected,
    x='x',
    y='y',
    hue='sig_mp',
    style='sig_ep',
    palette='deep',
    # s=s,
    ax=ax[2])
[x.plot([0, 0], [0, 12], '--k', alpha=0.5) for x in ax.flatten()]
# [x.set_xlim(-4, 4) for x in ax.flatten()]
plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datatable as dt
from scipy.interpolate import interp1d

fn = '../data/G345_adaptation.csv'

d = dt.fread(fn)

dd = d.to_numpy()
dd = np.reshape(dd, (-1, 2))

dd = dt.Frame(dd)
dd.names = ('x', 'y')
dd['trial'] = np.tile(np.arange(0, 1080, 1), dd.shape[0] // 1080)
dd['subject'] = np.repeat(np.arange(1, 21, 1), dd.shape[0] // 20)

samp_rate = 0.01
ns = 200

for i in range(0, d.shape[1], 2):

    x = d[:, i].to_numpy()[:, 0]
    y = d[:, i + 1].to_numpy()[:, 0]

    t = np.linspace(0, x.size * samp_rate, x.size)
    ts = np.linspace(t[0], t[-1], ns)

    fx = interp1d(t, x, kind='cubic')
    xs = fx(ts)

    fy = interp1d(t, y, kind='cubic')
    ys = fy(ts)

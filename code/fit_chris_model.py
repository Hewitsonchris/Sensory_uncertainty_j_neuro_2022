import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution
from scipy.optimize import curve_fit
from scipy.stats import ttest_1samp
from scipy.stats import sem
from scipy.stats import norm


def load_all_data():
    d0 = pd.read_csv('../data/exp1_2020.csv')
    d345 = pd.read_csv('../data/exp4_2020.csv')
    d13 = pd.read_csv('../data/exp7_2020.csv')
    d14 = pd.read_csv('../data/exp8_2020.csv')

    d = pd.concat((d0, d345, d13, d14), sort=False)
    return d

    d = load_all_data()

    dd = d.groupby(['GROUP', 'TRIAL_ABS',
                    'SIG_MP'])[['HA_INIT', 'HA_MID', 'HA_END',
                                'ROT']].mean().reset_index()


def fit_boot(group, bounds, maxiter, polish, n_boot_samp):

    d = load_all_data()

    for grp in group:

        dd = d[d['GROUP'] == grp]
        dd.sort_values('TRIAL_ABS', inplace=True)

        for n in range(n_boot_samp):

            boot_subs = np.random.choice(d['SUBJECT'].unique(),
                                         size=d['SUBJECT'].unique().shape[0],
                                         replace=True)

            ddd = []
            for i in range(boot_subs.shape[0]):

                ddd.append(dd[dd['SUBJECT'] == boot_subs[i]][[
                    'ROT', 'SIG_MP', 'SIG_EP', 'HA_INIT', 'HA_END',
                    'TRIAL_ABS', 'GROUP'
                ]])

            ddd = pd.concat(ddd)
            ddd = ddd.groupby('TRIAL_ABS').mean().reset_index()
            ddd.sort_values('TRIAL_ABS', inplace=True)

            rot = ddd.ROT.values
            sig_mp = ddd.SIG_MP.values
            sig_ep = ddd.SIG_EP.values
            group = d.GROUP.values
            x_obs_mp = ddd['HA_INIT'].values
            x_obs_ep = ddd['HA_END'].values

            n_trials = rot.shape[0]

            # fit power curve
            popt, pcov = curve_fit(f=exponential,
                                   xdata=n_trials,
                                   ydata=x_obs_mp,
                                   p0=[0, 0, 0])

            a = popt[0]
            b = popt[1]
            c = popt[2]

            curve = exponential(x_obs_mp, a, b, c)

            args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep, group, curve)

            results = differential_evolution(func=obj_func,
                                             bounds=bounds,
                                             args=args,
                                             disp=True,
                                             maxiter=maxiter,
                                             tol=1e-4,
                                             polish=polish,
                                             updating='deferred',
                                             workers=-1)

            fname = '../fits/chris_model/fit_' + str(grp) + '_boot.txt'
            with open(fname, 'a') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def obj_func(params, *args):

    w_init_1 = params[0]
    w_init_2 = params[1]
    w_init_3 = params[2]
    w_init_4 = params[3]
    w_end_1 = params[4]
    w_end_2 = params[5]
    w_end_3 = params[6]
    w_end_4 = params[7]
    rot = args[0]
    sig_mp = args[1]
    sig_ep = args[2]
    x_obs_mp = args[3]
    x_obs_ep = args[4]
    group = args[5]
    curve = args[6]

    pred = simulate(params, args)
    x_pred_mp = pred[0]
    x_pred_ep = pred[1]

    sse_mp = 100 * np.sum((x_obs_mp[:100] - x_pred_mp[:100])**2)
    sse_ep = 100 * np.sum((x_obs_ep[:100] - x_pred_ep[:100])**2)
    sse_mp += np.sum((x_obs_mp[100:] - x_pred_mp[100:])**2)
    sse_ep += np.sum((x_obs_ep[100:] - x_pred_ep[100:])**2)

    sse = sse_mp + sse_ep

    return sse


def simulate(params, args):

    w_init_1 = params[0]
    w_init_2 = params[1]
    w_init_3 = params[2]
    w_init_4 = params[3]
    w_end_1 = params[4]
    w_end_2 = params[5]
    w_end_3 = params[6]
    w_end_4 = params[7]
    rot = args[0]
    sig_mp = args[1]
    sig_ep = args[2]
    x_obs_mp = args[3]
    x_obs_ep = args[4]
    group = args[5]
    curve = args[6]

    n_trials = rot.shape[0]

    x_init_mod = np.zeros(n_trials)
    x_end_mod = np.zeros(n_trials)

    # modelling initial movement vector on trial t+1

    for i in range(1, n_trials):

        if sig_ep[i - 1] == 1:
            x_init_mod[i] = w_init_1 * curve[i]
        if sig_ep[i - 1] == 2:
            x_init_mod[i] = w_init_2 * curve[i]
        if sig_ep[i - 1] == 3:
            x_init_mod[i] = w_init_3 * curve[i]
        else:
            x_init_mod[i] = w_init_4 * x_init_mod[i - 1]

    # modelling endpoint on trial t

        if sig_mp[i] == 1:
            x_end_mod[i] = w_end_1 * (-rot[i] + x_init_mod[i])
        if sig_mp[i] == 2:
            x_end_mod[i] = w_end_2 * (-rot[i] + x_init_mod[i])
        if sig_mp[i] == 3:
            x_end_mod[i] = w_end_3 * (-rot[i] + x_init_mod[i])
        else:
            x_end_mod[i] = w_end_4 * x_init_mod[i]

            return (x_init_mod, x_end_mod)


bounds = ((-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5,
                                                                          5))
group = [13, 14, 3, 4, 5, 0]
maxiter = 1000
polish = False
n_boot_samp = 100
fit_boot(group, bounds, maxiter, polish, n_boot_samp)

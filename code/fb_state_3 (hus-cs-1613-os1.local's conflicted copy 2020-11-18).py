import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, LinearConstraint
from scipy.stats import ttest_1samp, linregress
from scipy.stats import sem
from scipy.stats import norm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pingouin as pg


def load_all_data():
    d1 = pd.read_csv('../datta/exp1_2020.csv')
    d2 = pd.read_csv('../datta/exp2_2020.csv')
    d3 = pd.read_csv('../datta/exp3_2020.csv')
    d4 = pd.read_csv('../datta/exp4_2020.csv')
    d5 = pd.read_csv('../datta/exp5_2020.csv')
    d6 = pd.read_csv('../datta/exp6_2020.csv')
    d7 = pd.read_csv('../datta/exp7_2020.csv')
    d8 = pd.read_csv('../datta/exp8_2020.csv')

    d = pd.concat((d1, d2, d3, d4, d5, d6, d7, d8), sort=False)

    d.PHASE = [x.lower() for x in d.PHASE.values]

    return d


def inspect_behaviour():
    # Group 0: sparse EP + no feedback washout
    # Group 1: no EP + no feedback washout
    # Group 2: all EP + 0 deg uniform washout
    # Group 3: KW rep + right hand transfer
    # Group 4: KW rep + left hand transfer
    # Group 5: KW rep + left hand transfer + opposite perturb
    # Group 6: Same as Group 0 + relearn + left hand 0 deg uniform
    # Group 7: unimodal -- uni likelihood -- (N=20, groups 7 and 8) no fb wash
    # Group 8: unimodal -- uni likelihood -- (N=20, groups 7 and 8) no fb wash
    # Group 9: bimodal predictable (N=20, groups 9 and10) + no fb wash
    # Group 10: bimodal predictable (N=20, groups 9 and10 + no fb wash
    # Group 11: bimodal stochastic (N=12, groups 11 and 12) + no fb wash
    # Group 12: bimodal stochastic (N=12, groups 11 and 12) + no fb wash

    d = load_all_data()

    # d[d['TRIAL_ABS'] < 100].groupby('SIG_MP')['HA_END', 'ROT'].plot(
    #     x='HA_END', y='ROT', kind='scatter')
    # plt.show()

    dd = d.groupby(['GROUP', 'TRIAL_ABS',
                    'SIG_MP'])[['HA_INIT', 'HA_MID', 'HA_END',
                                'ROT']].mean().reset_index()

    # dd1 = dd[dd['GROUP'] == 1][11:190]
    # d1 = dd1[dd1['SIG_MP'] == 1]['HA_INIT'].values
    # r1 = dd1[dd1['SIG_MP'] == 1]['ROT'].values
    # d2 = dd1[dd1['SIG_MP'] == 2]['HA_INIT'].values
    # r2 = dd1[dd1['SIG_MP'] == 2]['ROT'].values
    # d3 = dd1[dd1['SIG_MP'] == 3]['HA_INIT'].values
    # r3 = dd1[dd1['SIG_MP'] == 3]['ROT'].values
    # d4 = dd1[dd1['SIG_MP'] == 4]['HA_INIT'].values
    # r4 = dd1[dd1['SIG_MP'] == 4]['ROT'].values
    # e1 = r1 - d1
    # e2 = r2 - d2
    # e3 = r3 - d3
    # e4 = r4 - d4
    # # plt.plot(r1[0:-1], d1[1:], 'o')
    # # plt.plot(r2[0:-1], d2[1:], 'o')
    # # plt.plot(r3[0:-1], d3[1:], 'o')
    # # plt.plot(r4[0:-1], d4[1:], 'o')
    # plt.plot(e1, np.diff(d1, prepend=0), 'o')
    # plt.plot(e2, np.diff(d2, prepend=0), 'o')
    # plt.plot(e3, np.diff(d3, prepend=0), 'o')
    # plt.plot(e4, np.diff(d4, prepend=0), 'o')
    # # plt.legend(['SIG_MP = 1', 'SIG_MP = 2', 'SIG_MP = 3', 'SIG_MP = 4'])
    # plt.show()

    fig, ax = plt.subplots(1, 1)
    dd[dd['GROUP'] == 7].plot.scatter(x='TRIAL_ABS',
                                      y='HA_INIT',
                                      c='C0',
                                      marker='o',
                                      ax=ax)
    dd[dd['GROUP'] == 7].plot.scatter(x='TRIAL_ABS',
                                      y='HA_END',
                                      c='C0',
                                      marker='v',
                                      ax=ax)
    dd[dd['GROUP'] == 8].plot.scatter(x='TRIAL_ABS',
                                      y='HA_INIT',
                                      c='C1',
                                      marker='o',
                                      ax=ax)
    dd[dd['GROUP'] == 8].plot.scatter(x='TRIAL_ABS',
                                      y='HA_END',
                                      c='C1',
                                      marker='v',
                                      ax=ax)
    plt.show()

    dd[dd['GROUP'] == 1].plot.scatter(x='TRIAL_ABS',
                                      y='HA_INIT',
                                      c='SIG_MP',
                                      colormap='viridis')
    plt.show()

    # delta hand_angle as function of error size
    ha_ep = dd[dd['GROUP'] == 7]['HA_END'].values
    ha_mp = dd[dd['GROUP'] == 7]['HA_MID'].values
    ha_in = dd[dd['GROUP'] == 7]['HA_INIT'].values
    rot = dd[dd['GROUP'] == 7]['ROT'].values
    sig_mp = dd[dd['GROUP'] == 7]['SIG_MP'].values
    err_ha_mp = ha_mp - rot
    err_ha_ep = ha_ep - rot
    delta_ha_mp = np.diff(ha_mp, prepend=[0])
    delta_ha_ep = np.diff(ha_ep, prepend=[0])

    fig, ax = plt.subplots(2, 2)

    c = ['C0', 'C1', 'C2', 'C3', 'C4']
    for i in np.unique(sig_mp):
        ax[0, 0].plot(err_ha_mp[sig_mp == i],
                      delta_ha_mp[sig_mp == i],
                      'o',
                      alpha=0.5)
        ax[0, 0].set_xlabel('MP error size')
        ax[0, 0].set_ylabel('delta hand angle MP')

        ax[0, 1].plot(err_ha_mp[sig_mp == i],
                      delta_ha_ep[sig_mp == i],
                      'o',
                      alpha=0.5)
        ax[0, 1].set_xlabel('MP error size')
        ax[0, 1].set_ylabel('delta hand angle EP')

        ax[1, 0].plot(err_ha_ep[sig_mp == i],
                      delta_ha_mp[sig_mp == i],
                      'o',
                      alpha=0.5)
        ax[1, 0].set_xlabel('EP error size')
        ax[1, 0].set_ylabel('delta hand angle MP')

        ax[1, 1].plot(err_ha_ep[sig_mp == i],
                      delta_ha_ep[sig_mp == i],
                      'o',
                      alpha=0.5)
        ax[1, 1].set_xlabel('EP error size')
        ax[1, 1].set_ylabel('delta hand angle EP')

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(3, 5)
    g = 0
    for i in range(3):
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            ddd = dd[dd['GROUP'] == g]
            ax.plot(ddd['TRIAL_ABS'], ddd['HA_END'], '.', alpha=0.75)
            ax.plot(ddd['TRIAL_ABS'], ddd['HA_MID'], '.', alpha=0.75)
            ax.plot(ddd['TRIAL_ABS'], ddd['HA_INIT'], '.', alpha=0.75)
            ax.set_title('Group ' + str(ddd['GROUP'].unique()))
            ax.set_ylim([-30, 30])
            g += 1
    plt.tight_layout()
    plt.show()


def fit_validate(d, bounds, maxiter, polish, froot):

    for sub in d['SUBJECT'].unique():

        dd = d[d['SUBJECT'] == sub][[
            'ROT', 'HA_INIT', 'HA_END', 'TRIAL_ABS', 'GROUP', 'SIG_MP'
        ]]

        rot = d.ROT.values
        sig_mp = dd.SIG_MP.values
        group = d.GROUP.values
        x_obs_mp = d['HA_INIT'].values
        x_obs_ep = d['HA_END'].values
        args = (rot, sig_mp, x_obs_mp, x_obs_ep, group)

        p = np.loadtxt(froot + str(sub) + '.txt', delimiter=',')

        # simulate data from best fitting params
        (y, yff, yfb, xff, xfb) = simulate(p, args)

        args = (rot, sig_mp, yff, y, group)

        results = differential_evolution(func=obj_func,
                                         bounds=bounds,
                                         args=args,
                                         disp=True,
                                         maxiter=maxiter,
                                         tol=1e-15,
                                         polish=p,
                                         updating='deferred',
                                         workers=-1)

        fout = froot + str(sub) + '_val.txt'
        with open(fout, 'w') as f:
            tmp = np.concatenate((results['x'], [results['fun']]))
            tmp = np.reshape(tmp, (tmp.shape[0], 1))
            np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def inspect_fits_validate(d, froot):

    for sub in d['SUBJECT'].unique():

        pin = np.loadtxt(froot + str(sub) + '.txt', delimiter=',')
        pout = np.loadtxt(froot + str(sub) + '_val.txt', delimiter=',')

        pin = pin[:, :-1]
        pout = pout[:, :-1]

        names = [
            'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'w', 'gamma_ff',
            'gamma_fb', 'gamma_fbint', 'xfb_init'
        ]

        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        ax = ax.flatten()
        for j in range(pin.shape[1]):

            ax[j].plot(pin[:, j], pout[:, j], '.')
            ax[j].plot([-1, 1], [-1, 1], '--k', alpha=0.5)
            ax[j].set_title(names[j])

        plt.tight_layout()
        # plt.show()
        plt.savefig('../figures/fit_val_' + str(i) + '.pdf')


def inspect_fits_individual_model_compare_window(d):

    n_windows = 5
    n_params = 10

    subs = d['SUBJECT'].unique()

    xlabel = [
        'All nonzero', 'Zero FF learning', 'Zero FB learning',
        'Zero FB control', 'Nonzero FF Learning', 'Nonzero FB learning',
        'Nonzero FB control', 'All zero'
    ]

    n_models = len(xlabel)
    print(n_models)
    aic = np.zeros((n_models, subs.shape[0]))
    bic = np.zeros((n_models, subs.shape[0]))
    k_list = [9, 8, 8, 8, 7, 7, 7, 6]
    for win in range(n_windows):
        for i in range(n_models):

            froot = '../fits/fit_kw_adapt_' + str(i) + '_' + str(win) + '_'

            prec = np.zeros((subs.shape[0], n_params))

            for s in range(subs.shape[0]):
                dd = d[d['SUBJECT'] == s + 1]

                x_obs_mp = dd['HA_INIT'].values
                x_obs_ep = dd['HA_END'].values
                rot = dd['ROT'].values
                sig_mp = dd['SIG_MP'].values
                group = dd['GROUP'].values
                args = (rot, sig_mp, group)

                fname = froot + str(s + 1) + '.txt'
                p = np.loadtxt(fname, delimiter=',')
                prec[s, :] = p
                (y, yff, yfb, xff, xfb) = simulate(p[:-1], args)
                ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp))**2)
                ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp))**2)
                ss_res_mp = np.nansum((x_obs_mp - yff)**2)
                ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep))**2)
                ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep))**2)
                ss_res_ep = np.nansum((x_obs_ep - y)**2)
                r_squared = 1 - (ss_res_ep + ss_res_mp) / (ss_tot_ep +
                                                           ss_tot_mp)

                n = 1080
                k = k_list[i]
                bic[i, s] = compute_bic(r_squared, n, k)

            pname = [
                'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'w', 'gamma_ff',
                'gamma_fb', 'gamma_fbint', 'xfb_init', 'sse'
            ]

            # x = np.arange(1, n_params, 1)
            # plt.plot([1, n_params], [0, 0], '--')
            # plt.violinplot(prec[:, :-1])
            # plt.xticks(x, pname[:-1])
            # for jj in range(prec.shape[0]):
            #     plt.plot(x, prec[jj, :-1], '.', alpha=0.5)
            # plt.show()

            tstat, pval = ttest_1samp(prec, popmean=0, axis=0)
            cd = np.mean(prec, axis=0) / np.std(prec, axis=0, ddof=1)

            # print(('model: ' + str(i) + ' window: ' + str(win)))
            # inds = [5, 6, 7, 9]
            # for j in inds:
            #     print(pname[j] + ' = ' + str(np.round(prec[:, j].mean(), 2)) +
            #         ': t(' + str(prec.shape[0] - 1) + ') = ' +
            #         str(np.round(tstat[j], 2)) + ', p = ' +
            #         str(np.round(pval[j], 2)) + ', d = ' +
            #         str(np.round(cd[j], 2)))

        summed_bic = bic.sum(1)
        summed_bic = summed_bic - summed_bic[0]

        pbic = np.zeros(n_models)
        for i in range(n_models):
            pbic[i] = np.exp(-0.5 * summed_bic[i]) / np.sum(
                np.exp(-0.5 * summed_bic))

        # print(summed_bic.shape)
        # print(pbic)
        # print(pbic.sum())

        # fig, ax = plt.subplots(1, 2)
        # x = np.arange(1, 8, 1)
        # ax[0].plot(x, summed_bic, 'o', alpha=1)
        # ax[0].set_xticks(x)
        # ax[0].set_xticklabels(xlabel, rotation=30)
        # ax[0].set_ylabel('Summed BIC')
        # ax[1].plot(x, pbic, 'o', alpha=1)
        # ax[1].set_xticks(x)
        # ax[1].set_xticklabels(xlabel, rotation=30)
        # ax[1].set_ylabel('P(Model | Data)')
        # plt.show()

        pbic = np.zeros((n_models, subs.shape[0]))
        for s in range(subs.shape[0]):
            for i in range(n_models):
                pbic[i, s] = np.exp(-0.5 * bic[i, s]) / np.sum(
                    np.exp(-0.5 * bic[:, s]))

        b = pd.DataFrame(pbic.T)
        b.plot(kind='bar')
        plt.show()


def prepare_fit_summary(d):

    k_list = [9, 8, 8, 8, 7, 7, 7, 6]
    k_list = [x - 1 for x in k_list]
    n_models = len(k_list)

    drec = {
        'group': [],
        'subject': [],
        'model': [],
        'params': [],
        'x_obs_ep': [],
        'x_obs_mp': [],
        'sig_mp': [],
        'sig_ep': [],
        'rot': [],
        'y': [],
        'yff': [],
        'r_squared_ep': [],
        'r_squared_mp': [],
        'r_squared': [],
        'bic': []
    }

    for g, grp in enumerate(d['GROUP'].unique()):

        subs = d[d['GROUP'] == grp]['SUBJECT'].unique()

        for s, sub in enumerate(subs):

            for i in range(n_models):

                dd = d[(d['GROUP'] == grp) & (d['SUBJECT'] == sub)]
                x_obs_mp = dd['HA_INIT'].values
                x_obs_ep = dd['HA_END'].values
                rot = dd['ROT'].values
                sig_mp = dd['SIG_MP'].values
                sig_ep = dd['SIG_EP'].values
                group = dd['GROUP'].values
                args = (rot, sig_mp, sig_ep, group)

                fname = '../fits/fit_kw_adapt_' + str(i) + '_' + str(
                    grp) + '_' + str(sub) + '.txt'
                p = np.loadtxt(fname, delimiter=',')
                (y, yff, yfb, xff, xfb) = simulate(p[:-1], args)

                ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp))**2)
                ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp))**2)
                ss_res_mp = np.nansum((x_obs_mp - yff)**2)
                ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep))**2)
                ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep))**2)
                ss_res_ep = np.nansum((x_obs_ep - y)**2)

                r_squared_mp = 1 - (ss_res_mp) / (ss_tot_mp)
                r_squared_ep = 1 - (ss_res_ep) / (ss_tot_ep)
                r_squared = 1 - (ss_res_ep + ss_res_mp) / (ss_tot_ep +
                                                           ss_tot_mp)

                n = dd.shape[0]
                k = k_list[i]
                bic = compute_bic(r_squared, n, k)

                drec['group'].append(grp)
                drec['subject'].append(sub)
                drec['model'].append(i)
                drec['params'].append(p)
                drec['x_obs_ep'].append(x_obs_ep)
                drec['x_obs_mp'].append(x_obs_mp)
                drec['sig_mp'].append(sig_mp)
                drec['sig_ep'].append(sig_ep)
                drec['rot'].append(rot)
                drec['y'].append(y)
                drec['yff'].append(yff)
                drec['r_squared_ep'].append(r_squared_ep)
                drec['r_squared_mp'].append(r_squared_mp)
                drec['r_squared'].append(r_squared)
                drec['bic'].append(bic)

    drec = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in drec.items()]))

    drec.loc[drec['group'] == 3, 'group'] = 345
    drec.loc[drec['group'] == 4, 'group'] = 345
    drec.loc[drec['group'] == 5, 'group'] = 345

    return drec


def compute_pbic(x):
    x = x.values
    pbic = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        pbic[i] = np.exp(-0.5 * x[i]) / np.sum(np.exp(-0.5 * x[:]))
    return pbic


def report_fit_summary(d):

    d = prepare_fit_summary(d)

    # NOTE: perform model selection by posterior probability
    d['pbic'] = d.groupby(['group', 'subject'])['bic'].transform(compute_pbic)

    d['pbic_max'] = d.groupby(['group', 'subject'
                               ])['pbic'].transform(lambda x: np.max(x.values))

    xlabel = np.array([
        'All', 'No FF', 'No FB', 'No FB Int', 'Only FF', 'Only FB',
        'Only FB Int', 'None'
    ])

    d['model'] = xlabel[d['model'].values]

    d['best_model'] = d.groupby(
        ['group',
         'subject'])['pbic'].transform(lambda x: xlabel[np.argmax(x.values)])

    # NOTE: summarise model selection results
    fit_summary = d.loc[d['model'] == d['best_model']].groupby(
        ['group', 'best_model'])

    print(fit_summary['pbic', 'r_squared'].mean())
    print(fit_summary['best_model'].count())

    # NOTE: Report parameter estimate stats
    xlabel = np.array([
        'All', 'No FF', 'No FB', 'No FB Int', 'Only FF', 'Only FB',
        'Only FB Int', 'None'
    ])

    pname = np.array([
        'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'w', 'gamma_ff',
        'gamma_fb', 'gamma_fbint', 'xfb_init', 'sse'
    ])

    for g, grp in enumerate(d['group'].unique()):

        print('\n')
        print('group ' + str(grp))

        params = d.loc[d['model'] == 'All', ['group', 'subject', 'params']]
        params = np.vstack(
            params.loc[(params['group'] == grp), 'params'].values)

        tstat, pval = ttest_1samp(params, popmean=0, axis=0)
        cd = np.mean(params, axis=0) / np.std(params, axis=0, ddof=1)

        inds = [4, 5, 6, 7, 9]
        for j in inds:
            print(pname[j] + ' = ' + str(np.round(params[:, j].mean(), 2)) +
                  ': t(' + str(params.shape[0] - 1) + ') = ' +
                  str(np.round(tstat[j], 2)) + ', p = ' +
                  str(np.round(pval[j], 2)) + ', d = ' +
                  str(np.round(cd[j], 2)))


def get_slopes(x):
    group = x['GROUP'].iloc[0]
    subject = x['SUBJECT'].iloc[0]
    sig_ep = x['SIG_EP'].to_numpy()
    sig_mp = x['SIG_MP'].to_numpy()

    rot = x['ROT'].to_numpy()
    ep = x['HA_END'].to_numpy()
    mp = x['HA_MID'].to_numpy()

    error_mp = mp + rot
    error_ep = ep + rot
    fb_int = ep - mp
    delta_mp = np.diff(mp, append=0)

    d = {
        'group': [],
        'subject': [],
        'sig_ep': [],
        'sig_mp': [],
        'slope': [],
        'intercept': [],
        'process': []
    }

    for smp in np.unique(sig_mp):
        for sep in np.unique(sig_ep):

            X = error_mp[(sig_mp == smp) & (sig_ep == sep)][:, None]
            y = fb_int[(sig_mp == smp) & (sig_ep == sep)]
            if y.size > 0:
                regr = linear_model.LinearRegression().fit(X, y)
                slope = regr.coef_[0]
                inter = regr.intercept_
                d['group'].append(group)
                d['subject'].append(subject)
                d['sig_ep'].append(sep)
                d['sig_mp'].append(smp)
                d['slope'].append(slope)
                d['intercept'].append(inter)
                d['process'].append('fb')

            X = error_ep[(sig_mp == smp) & (sig_ep == sep)][:, None]
            y = delta_mp[(sig_mp == smp) & (sig_ep == sep)]
            if y.size > 0:
                regr = linear_model.LinearRegression().fit(X, y)
                slope = regr.coef_[0]
                inter = regr.intercept_
                d['group'].append(group)
                d['subject'].append(subject)
                d['sig_ep'].append(sep)
                d['sig_mp'].append(smp)
                d['slope'].append(slope)
                d['intercept'].append(inter)
                d['process'].append('ffep')

            X = error_mp[(sig_mp == smp) & (sig_ep == sep)][:, None]
            y = delta_mp[(sig_mp == smp) & (sig_ep == sep)]
            if y.size > 0:
                regr = linear_model.LinearRegression().fit(X, y)
                slope = regr.coef_[0]
                inter = regr.intercept_
                d['group'].append(group)
                d['subject'].append(subject)
                d['sig_ep'].append(sep)
                d['sig_mp'].append(smp)
                d['slope'].append(slope)
                d['intercept'].append(inter)
                d['process'].append('ffmp')

    return pd.DataFrame(d)


def report_slopes(x):
    res = pg.ttest(x['slope'], 1)

    rep = 'group = ' + str(x['group'].iloc[0])
    rep += ' process = ' + str(x['process'].iloc[0])
    rep += ', sig_ep = ' + str(x['sig_ep'].iloc[0])
    rep += ', sig_mp =  ' + str(x['sig_mp'].iloc[0])
    rep += ', slope = ' + str(x['slope'].iloc[0].round(2))
    rep += ', p = ' + str(res['p-val'].iloc[0].round(2))

    print(rep)


def plot_slopes_scatter(d):

    ncols = d['GROUP'].unique().size
    nrows = 3

    fig, ax = plt.subplots(nrows, ncols, squeeze=False)

    cc = np.array(['C0', 'C1', 'C2', 'C3', 'C4'])

    for g, grp in enumerate(np.sort(d['GROUP'].unique())):

        d_grp = d.loc[d['GROUP'] == grp]

        rot = np.vstack(d_grp['ROT'].values[:]).flatten()
        ep = np.vstack(d_grp['HA_END'].values[:]).flatten()
        mp = np.vstack(d_grp['HA_MID'].values[:]).flatten()
        # mp = np.vstack(d_grp['HA_INIT'].values[:]).flatten()
        delta_mp = np.diff(mp, append=0)
        fb_int = ep - mp
        sig_mp = np.vstack(d_grp['SIG_MP'].values[:]).flatten()
        sig_ep = np.vstack(d_grp['SIG_EP'].values[:]).flatten()
        error_mp = mp + rot
        error_ep = ep + rot

        a = 0.1
        for smp in np.unique(sig_mp):
            for sep in np.unique(sig_ep):
                x = error_mp[(sig_mp == smp) & (sig_ep == sep)]
                y = fb_int[(sig_mp == smp) & (sig_ep == sep)]
                if y.size > 0:
                    slope, intercept, _, pvalue, _ = linregress(x, y)
                    lab = ' sig_mp = ' + str(smp) + ', sig_ep = ' + str(sep)
                    # lab += ', m=' + str(slope.round(2))
                    # lab += ', b=' + str(intercept.round(2))
                    # lab += ', p=' + str(pvalue.round(2))
                    ax[0, g].scatter(x, y, alpha=a)
                    ax[0, g].plot(x,
                                  intercept + slope * x,
                                  label=lab,
                                  linestyle='-')
                    ax[0, g].set_xlabel('MP Error')
                    ax[0, g].set_ylabel('FB Integration')
                    # ax[0, g].set_xticks([])
                    # ax[0, g].set_yticks([])
                    ax[0, g].set_title(str(grp))
                    ax[0, g].legend(loc='lower left')

                x = error_ep[(sig_mp == smp) & (sig_ep == sep)]
                y = delta_mp[(sig_mp == smp) & (sig_ep == sep)]
                x = x[:-1]
                y = y[:-1]
                if y.size > 0:
                    slope, intercept, _, pvalue, _ = linregress(x, y)
                    lab = ' sig_mp=' + str(smp) + ', sig_ep=' + str(sep)
                    # lab += ', m=' + str(slope.round(2))
                    # lab += ', b=' + str(intercept.round(2))
                    # lab += ', p=' + str(pvalue.round(2))
                    ax[1, g].scatter(x, y, alpha=a)
                    ax[1, g].plot(x,
                                  intercept + slope * x,
                                  label=lab,
                                  linestyle='-')
                    ax[1, g].set_xlabel('EP Error')
                    ax[1, g].set_ylabel('FF Adaptation')
                    ax[1, g].set_xticks([])
                    ax[1, g].set_yticks([])
                    ax[1, g].set_title('Group ' + str(grp))
                    ax[1, g].legend(loc='lower left')

                x = error_mp[(sig_mp == smp) & (sig_ep == sep)]
                y = delta_mp[(sig_mp == smp) & (sig_ep == sep)]
                x = x[:-1]
                y = y[:-1]
                if y.size > 0:
                    slope, intercept, _, pvalue, _ = linregress(x, y)
                    lab = ' sig_mp = ' + str(smp) + ', sig_ep = ' + str(sep)
                    # lab += ', m=' + str(slope.round(2))
                    # lab += ', b=' + str(intercept.round(2))
                    # lab += ', p=' + str(pvalue.round(2))
                    ax[2, g].scatter(x, y, alpha=a)
                    ax[2, g].plot(x,
                                  intercept + slope * x,
                                  label=lab,
                                  linestyle='-')
                    ax[2, g].set_xlabel('MP Error')
                    ax[2, g].set_ylabel('FF Adaptation')
                    # ax[2, g].set_xticks([])
                    # ax[2, g].set_yticks([])
                    # ax[2, g].set_title('Group ' + str(grp))
                    ax[2, g].legend(loc='lower left')

    # plt.tight_layout()
    plt.show()


def inspect_fits_individual_model_compare(d):

    # NOTE: report fit summary
    # report_fit_summary(d)

    # NOTE: compute and report slopes
    d.loc[d['GROUP'] == 3, 'GROUP'] = 345
    d.loc[d['GROUP'] == 4, 'GROUP'] = 345
    d.loc[d['GROUP'] == 5, 'GROUP'] = 345

    dd = d.groupby(['GROUP', 'SUBJECT', 'SIG_MP',
                    'SIG_EP']).apply(get_slopes).reset_index(drop=True)

    # dd.groupby(['group', 'process', 'sig_ep', 'sig_mp']).apply(report_slopes)

    # NOTE: plot slopes scatter
    plot_slopes_scatter(d)

    # NOTE: examine full model parameter estimation method
    # n_params = 10

    # xlabel = np.array([
    #     'All', 'No FF', 'No FB', 'No FB Int', 'Only FF', 'Only FB',
    #     'Only FB Int', 'None'
    # ])

    # pname = np.array([
    #     'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'w', 'gamma_ff',
    #     'gamma_fb', 'gamma_fbint', 'xfb_init', 'sse'
    # ])

    # n_models = len(xlabel)

    # # nrows = 1
    # # ncols = d_fit_summary['group'].unique().shape[0]

    # fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    # for g, grp in enumerate(d_fit_summary['group'].unique()):

    #     params = d_fit_summary.loc[d_fit_summary['model'] ==
    #                                'All', ['group', 'subject', 'params']]
    #     params = np.vstack(
    #         params.loc[(params['group'] == grp), 'params'].values)

    #     x = np.arange(1, params.shape[1], 1)
    #     ax[0, g].plot([1, n_params], [0, 0], '--')
    #     ax[0, g].violinplot(params[:, :-1])
    #     # ax[0, g].set_xticks(x, pname[:-1])
    #     # ax[0, g].set_title(grp)
    #     for jj in range(params.shape[0]):
    #         ax[0, g].plot(x, params[jj, :-1], '.', alpha=0.5)

    #     tstat, pval = ttest_1samp(params, popmean=0, axis=0)
    #     cd = np.mean(params, axis=0) / np.std(params, axis=0, ddof=1)

    #     print('\n')
    #     print('group ' + str(grp))
    #     inds = [4, 5, 6, 7, 9]
    #     for j in inds:
    #         print(pname[j] + ' = ' + str(np.round(params[:, j].mean(), 2)) +
    #               ': t(' + str(params.shape[0] - 1) + ') = ' +
    #               str(np.round(tstat[j], 2)) + ', p = ' +
    #               str(np.round(pval[j], 2)) + ', d = ' +
    #               str(np.round(cd[j], 2)))

    # # plt.tight_layout()
    # # plt.show()

    # # TODO: for each best fit model, plot mean observed vs mean predicted
    # for g, grp in enumerate(d_fit_summary['group'].unique()):
    #     d_grp = d_fit_summary.loc[d_fit_summary['group'] == grp]
    #     grp_subs = d_grp['subject'].unique()
    #     nrow = 2
    #     ncol = grp_subs.size
    #     fig, ax = plt.subplots(nrow, ncol, figsize=(10, 6))
    #     for s, sub in enumerate(grp_subs):
    #         d_sub = d_grp.loc[(d_grp['subject'] == sub)
    #                           & (d_grp['model'] == d_grp['best_model'])]
    #         mp = d_sub['x_obs_mp'].to_numpy()[0]
    #         ep = d_sub['x_obs_mp'].to_numpy()[0]
    #         yff = d_sub['yff'].to_numpy()[0]
    #         y = d_sub['y'].to_numpy()[0]

    #         ax[0, s].plot(mp, '.C0')
    #         ax[0, s].plot(yff, '.C1')
    #         ax[1, s].plot(ep, '.C0')
    #         ax[1, s].plot(y, '.C1')
    #     plt.show()

    # s2 = d_fit_summary.loc[d_fit_summary['model'] == d_fit_summary['best_model'],
    #               ['group', 'best_model', 'x_obs_ep', 'x_obs_mp', 'y', 'yff']]

    # s2 = s2.melt(id_vars=['group', 'best_model'],
    #              value_vars=['x_obs_ep', 'x_obs_mp', 'y', 'yff'])

    # s3 = s2.groupby(['group', 'best_model', 'variable'
    #                  ]).apply(lambda x: np.vstack(x.values).mean(0)[0])

    # s3 = s3.to_frame('values').reset_index()

    # s4 = s3.pivot(index=None, columns='variable', values='values')

    # n_groups = s3['group'].unique().shape[0]
    # n_best_models = s3['best_model'].unique().shape[0]
    # fig, ax = plt.subplots(n_groups, n_best_models)
    # for g, grp, in enumerate(s3['group'].unique()):
    #     for m, model in enumerate(s3['best_model'].unique()):
    #         ep = s3.loc[(s3['group'] == grp) & (s3['best_model'] == model) &
    #                     (s3['variable'] == 'x_obs_ep'), 'values'].values
    #         mp = s3.loc[(s3['group'] == grp) & (s3['best_model'] == model) &
    #                     (s3['variable'] == 'x_obs_mp'), 'values'].values
    #         if ep.size > 0:
    #             ep = ep[0]
    #             mp = mp[0]

    #         ax[g, m].plot(ep, '.C0')
    #         ax[g, m].plot(mp, '.C1')
    #         ax[g, m].get_xaxis().set_ticks([])
    #         ax[g, m].get_yaxis().set_ticks([])
    #         ax[g, m].set_title(grp)
    # plt.show()

    # TODO: for each best fit model, validate the fit parameters and selection


def inspect_fits_boot(group):

    d = load_all_data()


def fit_individual(d, fit_args, froot):

    obj_func = fit_args['obj_func']
    bounds = fit_args['bounds']
    maxiter = fit_args['maxiter']
    disp = fit_args['disp']
    tol = fit_args['tol']
    polish = fit_args['polish']
    updating = fit_args['updating']
    workers = fit_args['workers']
    popsize = fit_args['popsize']
    mutation = fit_args['mutation']
    recombination = fit_args['recombination']

    for grp in d['GROUP'].unique():
        for sub in d[d['GROUP'] == grp]['SUBJECT'].unique():

            dd = d[(d['SUBJECT'] == sub) & (d['GROUP'] == grp)][[
                'ROT', 'HA_INIT', 'HA_END', 'TRIAL_ABS', 'GROUP', 'SIG_MP',
                'SIG_EP'
            ]]

            rot = dd.ROT.values
            sig_mp = dd.SIG_MP.values
            sig_ep = dd.SIG_EP.values
            group = dd.GROUP.values
            x_obs_mp = dd['HA_INIT'].values
            x_obs_ep = dd['HA_END'].values

            args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep, group)

            results = differential_evolution(func=obj_func,
                                             bounds=bounds,
                                             args=args,
                                             disp=disp,
                                             maxiter=maxiter,
                                             popsize=popsize,
                                             mutation=mutation,
                                             recombination=recombination,
                                             tol=tol,
                                             polish=polish,
                                             updating=updating,
                                             workers=workers)

            fout = froot + str(grp) + '_' + str(sub) + '.txt'
            with open(fout, 'w') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


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
                    'ROT', 'SIG_MP', 'HA_INIT', 'HA_END', 'TRIAL_ABS', 'GROUP'
                ]])

            ddd = pd.concat(ddd)
            ddd = ddd.groupby('TRIAL_ABS').mean().reset_index()
            ddd.sort_values('TRIAL_ABS', inplace=True)

            rot = ddd.ROT.values
            sig_mp = ddd.SIG_MP.values
            group = ddd.GROUP.values
            x_obs_mp = ddd['HA_INIT'].values
            x_obs_ep = ddd['HA_END'].values

            args = (rot, sig_mp, x_obs_mp, x_obs_ep, group)

            results = differential_evolution(func=obj_func,
                                             bounds=bounds,
                                             args=args,
                                             disp=True,
                                             maxiter=maxiter,
                                             tol=1e-15,
                                             polish=p,
                                             updating='deferred',
                                             workers=-1)

            fname = '...fits/fit_' + str(grp) + '_boot.txt'
            with open(fname, 'a') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def obj_func(params, *args):

    obs = args

    rot = obs[0]
    sig_mp = obs[1]
    sig_ep = obs[2]
    x_obs_mp = obs[3]
    x_obs_ep = obs[4]
    group = obs[5]

    args = (rot, sig_mp, sig_ep, group)

    x_pred = simulate(params, args)
    x_pred_mp = x_pred[1]
    x_pred_ep = x_pred[0]

    # sse_mp = 100 * np.sum((x_obs_mp[:100] - x_pred_mp[:100])**2)
    # sse_ep = 100 * np.sum((x_obs_ep[:100] - x_pred_ep[:100])**2)
    # sse_mp += np.sum((x_obs_mp[100:] - x_pred_mp[100:])**2)
    # sse_ep += np.sum((x_obs_ep[100:] - x_pred_ep[100:])**2)

    sse_mp = np.sum((x_obs_mp - x_pred_mp)**2)
    sse_ep = np.sum((x_obs_ep - x_pred_ep)**2)

    sse = sse_mp + sse_ep

    return sse


def simulate(params, args):

    alpha_ff = params[0]
    beta_ff = params[1]
    alpha_fb = params[2]
    beta_fb = params[3]
    w = params[4]
    gamma_ff = params[5]
    gamma_fb = params[6]
    gamma_fbint = params[7]
    xfb_init = params[8]

    r = args[0]
    sig_mp = args[1]
    sig_ep = args[2]
    group = args[3]

    n_trials = r.shape[0]

    delta_ep = np.zeros(n_trials)
    delta_mp = np.zeros(n_trials)
    xff = np.zeros(n_trials)
    xfb = np.zeros(n_trials)
    yff = np.zeros(n_trials)
    yfb = np.zeros(n_trials)
    y = np.zeros(n_trials)

    xfb[0] = xfb_init

    wmp = w
    wep = 1 - w

    for i in range(n_trials - 1):

        bayes_mod_ff = 0.0
        bayes_mod_fb = 0.0
        bayes_mod_fbint = 0.0

        # start to midpoint
        yff[i] = xff[i]
        yfb[i] = 0.0
        y[i] = yff[i] + yfb[i]

        # midpoint to endpoint
        if sig_mp[i] != 4:
            # bayes_mod_ff_mp = bayes_int(sig_mp[i], gamma_ff)
            bayes_mod_fbint = bayes_int(sig_mp[i], gamma_fbint)
            delta_mp[i] = 0.0 - (y[i] + r[i])
            yfb[i] = xfb[i] * delta_mp[i] * bayes_mod_fbint

            ff_adapt_mp = wmp * alpha_ff * (gamma_ff * sig_mp[i] +
                                            (delta_mp[i] - gamma_ff))

        else:
            # bayes_mod_ff_mp = 0
            delta_mp[i] = 0.0
            yfb[i] = 0.0
            ff_adapt_mp = 0.0

        y[i] = yff[i] + yfb[i]

        if sig_ep[i] == 4:
            bayes_mod_ff_ep = bayes_int(sig_ep[i], gamma_ff)
            bayes_mod_fb = bayes_int(sig_mp[i], gamma_fb)
            delta_ep[i] = 0.0 - (y[i] + r[i])

            ff_adapt_ep = wep * alpha_ff * (gamma_ff * sig_ep[i] +
                                            (delta_ep[i] - gamma_ff))

        else:
            # bayes_mod_ff_ep = 0
            bayes_mod_fb = 0
            delta_ep[i] = 0.0
            ff_adapt_ep = 0.0

        # ff_adapt_mp = wmp * alpha_ff * delta_mp[i] * bayes_mod_ff_mp
        # ff_adapt_ep = wep * alpha_ff * delta_ep[i] * bayes_mod_ff_ep
        # xff[i + 1] = beta_ff * xff[i] + ff_adapt_mp + ff_adapt_ep

        xff[i + 1] = beta_ff * xff[i] + ff_adapt_mp + ff_adapt_ep

        xfb[i + 1] = beta_fb * xfb[i] + alpha_fb * delta_ep[i] * bayes_mod_fb

        xfb = np.clip(xfb, -2, 2)

    return (y, yff, yfb, xff, xfb)


def bayes_int(x, m):

    # x = np.arange(-1, 3, 0.01)
    # plt.plot(x, np.tanh(0 * (x - 2)) / 2 + 0.5, label=str(0))
    # plt.plot(x, np.tanh(1 * (x - 2)) / 2 + 0.5, label=str(1))
    # plt.plot(x, np.tanh(2 * (x - 2)) / 2 + 0.5, label=str(2))
    # plt.plot(x, np.tanh(3 * (x - 2)) / 2 + 0.5, label=str(3))
    # plt.ylim([-0.01, 1.01])
    # plt.xlim([1, 3])
    # plt.xticks([1, 2, 3])
    # plt.ylabel('f(x)')
    # plt.xlabel('sigma')
    # plt.legend()
    # plt.show()

    return np.tanh(m * (x - 2)) / 2 + 0.5


def bootstrap_ci(x, n, alpha):
    x_boot = np.zeros(n)
    for i in range(n):
        x_boot[i] = np.random.choice(x, x.shape, replace=True).mean()
        ci = np.percentile(x_boot, [alpha / 2, 1.0 - alpha / 2])
    return (ci)


def bootstrap_t(x_obs, y_obs, x_samp_dist, y_samp_dist, n):
    d_obs = x_obs - y_obs

    d_boot = np.zeros(n)
    xs = np.random.choice(x_samp_dist, n, replace=True)
    ys = np.random.choice(y_samp_dist, n, replace=True)
    d_boot = xs - ys
    d_boot = d_boot - d_boot.mean()

    p_null = (1 + np.sum(np.abs(d_boot) > np.abs(d_obs))) / (n + 1)
    return (p_null)


def compute_aic(rsq, n, k):
    # aic = 2 * k + n * np.log(sse / n)
    aic = n * np.log(1 - rsq) + k * 2
    return aic


def compute_bic(rsq, n, k):
    # bic = np.log(n) * k + n * np.log(sse / n)
    bic = n * np.log(1 - rsq) + k * np.log(n)
    return bic


nboot = -1
d = load_all_data()
# dd = d.set_index('GROUP', drop=False).loc[[0, 13, 14, 3, 4, 5, 7, 8]]
dd = d.set_index('GROUP', drop=False).loc[[13, 14]]
# dd = d.set_index('GROUP', drop=False).loc[[0, 3, 4, 5]]
# dd = d.set_index('GROUP', drop=False).loc[[7, 8]]
dd = dd.set_index('PHASE', drop=False).loc['adaptation']
dd = dd[dd['TRIAL'] <= 180]

all = pd.DataFrame({
    'model': 'all',
    'bid': ('lb', 'ub'),
    'alpha_ff': (0, 1),
    'beta_ff': (0, 1),
    'alpha_fb': (0, 1),
    'beta_fb': (0, 1),
    'w': (0, 1),
    'gamma_ff': (-100, 0),
    'gamma_fb': (0, 0),
    'gamma_fbint': (-3, 3),
    'xfb_init': (-2, 2)
})

only_ff = pd.DataFrame({
    'model': 'only_ff',
    'bid': ('lb', 'ub'),
    'alpha_ff': (0, 1),
    'beta_ff': (0, 1),
    'alpha_fb': (0, 1),
    'beta_fb': (0, 1),
    'w': (0, 1),
    'gamma_ff': (-100, 0),
    'gamma_fb': (0, 0),
    'gamma_fbint': (0, 0),
    'xfb_init': (-2, 2)
})

only_fbint = pd.DataFrame({
    'model': 'only_fbint',
    'bid': ('lb', 'ub'),
    'alpha_ff': (0, 1),
    'beta_ff': (0, 1),
    'alpha_fb': (0, 1),
    'beta_fb': (0, 1),
    'w': (0, 1),
    'gamma_ff': (0, 0),
    'gamma_fb': (0, 0),
    'gamma_fbint': (-3, 3),
    'xfb_init': (-2, 2)
})

b = pd.concat((all, only_ff, only_fbint))

for i, mod in enumerate(b.model.unique()):
    lb = b.loc[(b['model'] == mod) & (b['bid'] == 'lb')].drop(
        ['model', 'bid'], axis=1).to_numpy()[0]

    ub = b.loc[(b['model'] == mod) & (b['bid'] == 'ub')].drop(
        ['model', 'bid'], axis=1).to_numpy()[0]

    bb = tuple(zip(lb, ub))
    print(bb)

    # To improve your chances of finding a global minimum use higher popsize
    # values (default 15), with higher mutation (default 0.5) and (dithering),
    # but lower recombination (default 0.7) values. This has the effect of
    # widening the search radius, but slowing convergence.
    fit_args = {
        'obj_func': obj_func,
        'bounds': bb,
        'disp': False,
        'maxiter': 1000,
        'popsize': 20,
        'mutation': 0.7,
        'recombination': 0.5,
        'tol': 1e-6,
        'polish': False,
        'updating': 'deferred',
        'workers': -1
    }

    froot = '../fits/fit_kw_adapt_' + str(i) + '_'
    # fit_individual(dd, fit_args, froot)

inspect_fits_individual_model_compare(dd)

# fit_boot(dd, b, m, p, nboot)
# inspect_fits_boot()
# inspect_behaviour()

# dd.plot(x='GROUP', y='ROT', kind='scatter')
# plt.show()

# TODO: apply model comparison approach to w?
# TODO: fit via bootstrap
# TODO: fit via ML / EM
# TODO: model comparison might be too conservative (look more into fits)

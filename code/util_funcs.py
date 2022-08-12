from imports import *


def define_models():

    bnds = pd.DataFrame({
        'credit': 'smart',
        'n_params': 18,
        'bid': ('lb', 'ub'),
        'alpha_ff': (0, 1),
        'beta_ff': (0, 1),
        'bias_ff': (0, 0),
        'alpha_ff2': (0, 1),
        'beta_ff2': (0, 1),
        'bias_ff2': (-10, 10),
        'alpha_fb': (0, 1),
        'beta_fb': (-10, 10),
        'xfb_init': (-2, 2),
        'gamma_fbint_1': (0, 1),
        'gamma_fbint_2': (0, 1),
        'gamma_fbint_3': (0, 1),
        'gamma_fbint_4': (0, 1),
        'gamma_ff_1': (0, 1),
        'gamma_ff_2': (0, 1),
        'gamma_ff_3': (0, 1),
        'gamma_ff_4': (0, 1),
        'temporal_discount': (0, 1)
    })

    m0 = bnds.copy()
    m0['name'] = 'error-scale'

    m00 = bnds.copy()
    m00['name'] = 'error-scale-one-state'
    m00['alpha_ff'] = (0, 0)
    m00['beta_ff'] = (0, 0)
    m00['bias_ff'] = (0, 0)
    m00['n_params'] = 15

    m1 = bnds.copy()
    m1['name'] = 'state-scale'

    m11 = bnds.copy()
    m11['name'] = 'state-scale-one-state'
    m11['alpha_ff'] = (0, 0)
    m11['beta_ff'] = (0, 0)
    m11['bias_ff'] = (0, 0)
    m11['n_params'] = 15

    m2 = bnds.copy()
    m2['name'] = 'bias-scale'
    m2['gamma_ff_2'] = (-1, 1)
    m2['gamma_ff_3'] = (-1, 1)
    m2['gamma_ff_4'] = (-1, 1)

    m22 = bnds.copy()
    m22['name'] = 'bias-scale-one-state'
    m22['gamma_ff_2'] = (-1, 1)
    m22['gamma_ff_3'] = (-1, 1)
    m22['gamma_ff_4'] = (-1, 1)
    m22['alpha_ff'] = (0, 0)
    m22['beta_ff'] = (0, 0)
    m22['bias_ff'] = (0, 0)
    m22['n_params'] = 15

    b = pd.concat((m00, m11, m22, m0, m1, m2))

    return b


def fit_models(models, dd):

    b = models

    for i, modname in enumerate(b['name'].unique()):

        credit = b.loc[(b['name'] == modname), 'credit'][0]

        lb = b.loc[(b['name'] == modname) & (b['bid'] == 'lb')].drop(
            ['name', 'bid', 'n_params', 'credit'], axis=1).to_numpy()[0]

        ub = b.loc[(b['name'] == modname) & (b['bid'] == 'ub')].drop(
            ['name', 'bid', 'n_params', 'credit'], axis=1).to_numpy()[0]

        bb = tuple(zip(lb, ub))

        constraints = LinearConstraint(
            A=[[1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            lb=[-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ub=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # to improve your chances of finding a global minimum use higher
        # popsize (default 15), with higher mutation (default 0.5) and
        # (dithering), but lower recombination (default 0.7). this has the
        # effect of widening the search radius, but slowing convergence.
        fit_args = {
            'obj_func': obj_func,
            'bounds': bb,
            'constraints': constraints,
            'disp': False,
            'maxiter': 3000,
            'popsize': 20,
            'mutation': 0.7,
            'recombination': 0.5,
            'tol': 1e-3,
            'polish': False,
            'updating': 'deferred',
            'workers': -1
        }

        froot = '../fits/fit_' + modname
        fit_individual(modname, dd, fit_args, froot)
        # fit_boot(credit, dd, fit_args, froot)


def load_all_data():
    # d1 = pd.read_csv('../data/exp1_2020.csv')
    # d2 = pd.read_csv('../data/exp2_2020.csv')
    # d3 = pd.read_csv('../data/exp3_2020.csv')
    # d4 = pd.read_csv('../data/exp4_2020.csv')
    # d5 = pd.read_csv('../data/exp5_2020.csv')
    # d6 = pd.read_csv('../data/exp6_2020.csv')
    # d7 = pd.read_csv('../data/exp7_2020.csv')
    # d8 = pd.read_csv('../data/exp8_2021.csv')
    # d345 = pd.read_csv('../data/exp345_2021.csv')
    d15 = pd.read_csv('../data/G15.csv')          # group 15: unimodal stochastic (n=20, low uncertainty, high uncertainty [2x2], exp1 2022 paper 1)
    d16 = pd.read_csv('../data/G16.csv')
    d15['HA_INIT'] = d15['HA_INT']     
    d15['HA_MID'] = d15['HA_INT']
    d1718 = pd.read_csv('../data/G17_18.csv') 
    d1920 = pd.read_csv('../data/G19_20.csv')     # group 19: unimodal stochastic (n=20, low uncertainty, medium uncertainty, high uncertainty, no-fb, 'Midpoint only' exp2 2022 paper 1)
                                                  # group 20: unimodal stochastic (n=20, low uncertainty, medium uncertainty, high uncertainty, no-fb, 'Midpoint plus enpoint incongruent' exp3 2022 paper 1)
    


    d = pd.concat((d15, d16, d1718, d1920), sort=False)

    d.columns = d.columns.str.lower()
    d.phase = [x.lower() for x in d.phase.to_numpy()]

    return d


def inspect_behaviour(d):

    for grp in d.group.unique():
        dd = d.loc[np.isin(d['group'].to_numpy(), grp)]

        dd = dd.groupby(['group', 'phase', 'trial', 'sig_mp',
                         'sig_ep']).mean().reset_index()

        dd['delta_ha_mid'] = np.diff(dd['ha_mid'].to_numpy(), append=0)
        dd['fb_int'] = dd['ha_end'].to_numpy() - dd['ha_mid'].to_numpy()
        dd['error_mp'] = dd['ha_mid'].to_numpy() + dd['rot'].to_numpy()
        dd['error_ep'] = dd['ha_end'].to_numpy() + dd['rot'].to_numpy()

        dd['sig_mp'] = dd['sig_mp'].astype('category')
        dd['sig_ep'] = dd['sig_ep'].astype('category')

        fig, ax = plt.subplots(2, 2, squeeze=False)


        sns.violinplot(x='sig_mp',
                       y='delta_ha_mid',
                       hue='sig_ep',
                       data=dd,
                       ax=ax[0, 0])
        ax[0, 0].set_xlabel('sig mp')
        ax[0, 0].set_ylabel('delta ha mid')
        ax[0, 0].set_title('ff adapt')

        sns.violinplot(x='sig_mp',
                       y='fb_int',
                       hue='sig_ep',
                       data=dd,
                       ax=ax[0, 1])
        ax[0, 1].set_xlabel('sig mp')
        ax[0, 1].set_ylabel('ep - mp')
        ax[0, 1].set_title('fb int')

        if np.isin(grp, [13, 14]):
            sns.scatterplot(x='error_mp',
                            y='delta_ha_mid',
                            style='sig_mp',
                            hue='sig_ep',
                            legend='full',
                            data=dd,
                            ax=ax[1, 0])

            sns.scatterplot(x='error_mp',
                            y='fb_int',
                            style='sig_mp',
                            hue='sig_ep',
                            data=dd,
                            ax=ax[1, 1])

        else:
            sns.scatterplot(x='error_mp',
                            y='delta_ha_mid',
                            style='sig_ep',
                            hue='sig_mp',
                            legend='full',
                            data=dd,
                            ax=ax[1, 0])

            sns.scatterplot(x='error_mp',
                            y='fb_int',
                            style='sig_ep',
                            hue='sig_mp',
                            data=dd,
                            ax=ax[1, 1])

        ax[1, 0].set_xlabel('error mp')
        ax[1, 0].set_ylabel('delta ha mid')
        ax[1, 0].set_title('ff adapt')

        ax[1, 1].set_xlabel('error mp')
        ax[1, 1].set_ylabel('ep - mp')
        ax[1, 1].set_title('fb int')

        fig.suptitle('group = ' + str(grp), fontsize=14)

        plt.tight_layout()
        plt.show()


def inspect_behaviour_2(models, d):

    dfs = prepare_fit_summary(models, d)

    for grp in d.group.unique():

        print(grp)

        d_grp = dfs.loc[np.isin(dfs['group'].to_numpy(), grp)]

        for i in range(d_grp.shape[0]):

            dd = d_grp.iloc[i]

            yff = dd.yff
            y = dd.y
            x_obs_mp = dd.x_obs_mp
            x_obs_ep = dd.x_obs_ep

            fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 6))

            t = np.arange(0, y.shape[0], 1)
            ax[0, 0].plot(t, x_obs_mp, '-C0', label='observed')
            ax[0, 0].plot(t, yff, '-C1', label='predicted')

            ax[0, 1].plot(t, x_obs_ep, '-C0', label='observed')
            ax[0, 1].plot(t, y, '-C1', label='predicted')

            ax[0, 0].set_title('r-sq mp = ' + str(dd.r_squared_mp.round(2)))
            ax[0, 1].set_title('r-sq ep = ' + str(dd.r_squared_ep.round(2)))

            ax[0, 0].legend()
            ax[0, 1].legend()

            plt.tight_layout()
            plt.show()


def inspect_behaviour_all(d):
    # group 0: sparse ep + no feedback washout
    # group 1: no ep + no feedback washout
    # group 2: all ep + 0 deg uniform washout
    # group 3: kw rep + right hand transfer
    # group 4: kw rep + left hand transfer
    # group 5: kw rep + left hand transfer + opposite perturb
    # group 6: same as group 0 + relearn + left hand 0 deg uniform
    # group 7: unimodal -- uni likelihood -- (n=20, groups 7 and 8) no fb wash
    # group 8: unimodal -- uni likelihood -- (n=20, groups 7 and 8) no fb wash
    # group 9: bimodal predictable (n=20, groups 9 and10) + no fb wash
    # group 10: bimodal predictable (n=20, groups 9 and10 + no fb wash
    # group 11: bimodal stochastic (n=12, groups 11 and 12) + no fb wash
    # group 12: bimodal stochastic (n=12, groups 11 and 12) + no fb wash
    # group 15: unimodal stochastic (n=20, low uncertainty, high uncertainty [2x2], exp1 2022 paper 1)
    # group 19: unimodal stochastic (n=20, low uncertainty, medium uncertainty, high uncertainty, no-fb, 'Midpoint only' exp2 2022 paper 1)
    # group 20: unimodal stochastic (n=20, low uncertainty, medium uncertainty, high uncertainty, no-fb, 'Midpoint plus enpoint incongruent' exp3 2022 paper 1)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(3, 5)
    g = 0
    for i in range(3):
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            dd = d[d['group'] == g]
            ax.plot(dd['trial_abs'], dd['ha_end'], '.', alpha=0.75)
            ax.plot(dd['trial_abs'], dd['ha_mid'], '.', alpha=0.75)
            ax.plot(dd['trial_abs'], dd['ha_init'], '.', alpha=0.75)
            ax.set_title('group ' + str(dd['group'].unique()))
            ax.set_ylim([-30, 30])
            g += 1
    plt.tight_layout()
    plt.show()


def fit_validate(d, bounds, maxiter, polish, froot):

    for sub in d['subject'].unique():

        dd = d[d['subject'] == sub][[
            'rot', 'ha_init', 'ha_end', 'trial_abs', 'group', 'sig_mp'
        ]]

        rot = d.rot.to_numpy()
        sig_mp = dd.sig_mp.to_numpy()
        group = d.group.to_numpy()
        x_obs_mp = d['ha_init'].to_numpy()
        x_obs_ep = d['ha_end'].to_numpy()
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

    for sub in d['subject'].unique():

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


def prepare_fit_summary_boot(models, d):

    drec = {
        'group': [],
        'model': [],
        'params': [],
        'x_pred_ep': [],
        'x_pred_mp': [],
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

    for g, grp in enumerate(d['group'].unique()):

        d_grp = d[(d['group'] == grp)].groupby(['trial_abs']).mean()

        for i, modname in enumerate(models['name'].unique()):

            credit = models.loc[(models['name'] == modname), 'credit'][0]

            x_obs_mp = d_grp['ha_init'].to_numpy()
            x_obs_ep = d_grp['ha_end'].to_numpy()
            rot = d_grp['rot'].to_numpy()
            sig_mp = d_grp['sig_mp'].to_numpy().astype(np.int)
            sig_ep = d_grp['sig_ep'].to_numpy().astype(np.int)
            group = d_grp['group'].to_numpy()
            args = (rot, sig_mp, sig_ep, group, credit)

            fname = '../fits/fit_' + modname + '_group_' + str(
                grp) + '_boot.txt'

            p = np.loadtxt(fname, delimiter=',')
            print(grp, p.shape)

            fname = '../fits/fit_' + modname + '_group_' + str(
                grp) + '_grp.txt'

            p_grp = np.loadtxt(fname, delimiter=',')

            (y, yff, yfb, xff, xfb) = simulate(p_grp[:-1], args)

            # NOTE: must have programmed simulate to stop a trial early
            y = y[:-1]
            yff = yff[:-1]
            yfb = yfb[:-1]
            xff = xff[:-1]
            xfb = xfb[:-1]
            x_obs_mp = x_obs_mp[:-1]
            x_obs_ep = x_obs_ep[:-1]

            # fig, ax = plt.subplots(1, 3, squeeze=False)
            # ax[0, 0].violinplot(p[:, 0:5])
            # ax[0, 1].violinplot(p[:, 5:8])
            # ax[0, 2].violinplot(p[:, 8:11])
            # plt.show()

            fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(10, 10))
            ax[0, 0].plot(x_obs_mp)
            ax[0, 0].plot(yff)
            ax[0, 1].plot(x_obs_ep)
            ax[0, 1].plot(y)
            ax[1, 0].violinplot(p[:, :5], np.arange(0, 5, 1))
            ax[1, 1].violinplot(p[:, 5:8], np.arange(0, 3, 1))
            ax[1, 2].violinplot(p[:, 8:-1], np.arange(0, 3, 1))
            plt.title('group ' + str(grp) + ', model ' + modname)
            plt.savefig('../figures/fit_summary_grp_' + str(grp) + '_mod_' +
                        modname + '.pdf')
            # plt.show()

            ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp))**2)
            ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp))**2)
            ss_res_mp = np.nansum((x_obs_mp - yff)**2)
            ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep))**2)
            ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep))**2)
            ss_res_ep = np.nansum((x_obs_ep - y)**2)

            r_squared_mp = 1 - (ss_res_mp) / (ss_tot_mp)
            r_squared_ep = 1 - (ss_res_ep) / (ss_tot_ep)
            r_squared = 1 - (ss_res_ep + ss_res_mp) / (ss_tot_ep + ss_tot_mp)

            n = d_grp.shape[0]
            k = models.loc[models['name'] == modname, 'n_params'].unique()[0]
            bic = compute_bic(r_squared, n, k)

            drec['group'].append(grp)
            drec['model'].append(modname)
            drec['params'].append(p_grp)
            drec['x_pred_ep'].append(y)
            drec['x_pred_mp'].append(yff)
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

    return drec


def prepare_fit_summary(models, d):

    drec = {
        'group': [],
        'subject': [],
        'model': [],
        'params': [],
        'x_pred_ep': [],
        'x_pred_mp': [],
        'x_obs_ep': [],
        'x_obs_mp': [],
        'sig_mp': [],
        'sig_ep': [],
        'rot': [],
        'y': [],
        'yff': [],
        'xff': [],
        'xff2': [],
        'r_squared_ep': [],
        'r_squared_mp': [],
        'r_squared': [],
        'bic': []
    }

    for g, grp in enumerate(d['group'].unique()):

        subs = d[d['group'] == grp]['subject'].unique()

        for s, sub in enumerate(subs):

            for i, modname in enumerate(models['name'].unique()):

                credit = models.loc[(models['name'] == modname), 'credit'][0]

                dd = d[(d['group'] == grp) & (d['subject'] == sub)]
                x_obs_mp = dd['ha_init'].to_numpy()
                x_obs_ep = dd['ha_end'].to_numpy()
                rot = dd['rot'].to_numpy()
                sig_mp = dd['sig_mp'].to_numpy()
                sig_ep = dd['sig_ep'].to_numpy()
                group = dd['group'].to_numpy()
                args = (rot, sig_mp, sig_ep, group, modname)

                fname = '../fits/fit_' + modname + '_group_' + str(
                    grp) + '_sub_' + str(sub) + '.txt'
                p = np.loadtxt(fname, delimiter=',')
                (y, yff, yfb, xff, xfb, xff2) = simulate(p[:-1], args)

                x_obs_mp = x_obs_mp[:-1]
                x_obs_ep = x_obs_ep[:-1]
                yff = yff[:-1]
                xff = xff[:-1]
                xff2 = xff2[:-1]
                y = y[:-1]
                sig_mp = sig_mp[:-1]
                sig_ep = sig_ep[:-1]

                ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp))**2)
                ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp))**2)
                ss_res_mp = np.nansum((x_obs_mp - yff)**2)
                ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep))**2)
                ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep))**2)
                ss_res_ep = np.nansum((x_obs_ep - y)**2)

                r_squared_mp = 1 - ss_res_mp / ss_tot_mp
                r_squared_ep = 1 - ss_res_ep / ss_tot_ep
                r_squared = 1 - (ss_res_ep + ss_res_mp) / (ss_tot_ep +
                                                           ss_tot_mp)

                n = dd.shape[0]
                k = models.loc[models['name'] == modname,
                               'n_params'].unique()[0]
                bic = compute_bic(r_squared, n, k)

                drec['group'].append(grp)
                drec['subject'].append(sub)
                drec['model'].append(modname)
                drec['params'].append(p)
                drec['x_pred_ep'].append(y)
                drec['x_pred_mp'].append(yff)
                drec['x_obs_ep'].append(x_obs_ep)
                drec['x_obs_mp'].append(x_obs_mp)
                drec['sig_mp'].append(sig_mp)
                drec['sig_ep'].append(sig_ep)
                drec['rot'].append(rot)
                drec['y'].append(y)
                drec['yff'].append(yff)
                drec['xff'].append(xff)
                drec['xff2'].append(xff2)
                drec['r_squared_ep'].append(r_squared_ep)
                drec['r_squared_mp'].append(r_squared_mp)
                drec['r_squared'].append(r_squared)
                drec['bic'].append(bic)

    drec = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in drec.items()]))

    return drec


def compute_pbic(x):
    x = x.to_numpy()
    pbic = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        pbic[i] = np.exp(-0.5 * x[i]) / np.sum(np.exp(-0.5 * x[:]))
    return pbic


def report_fit_summary_boot(models, d):

    dd = prepare_fit_summary_boot(models, d)

    dd['pbic'] = dd.groupby(['group'])['bic'].transform(compute_pbic)

    dd['pbic_max'] = dd.groupby(
        ['group'])['pbic'].transform(lambda x: np.max(x.to_numpy()))

    fit_summary = dd.loc[dd['pbic'] == dd['pbic_max']].groupby(
        ['group', 'model'])
    fit_summary = dd.groupby(['group', 'model'])

    print()
    print(fit_summary['pbic', 'r_squared', 'r_squared_mp',
                      'r_squared_ep'].mean())
    print()
    print(fit_summary['model'].count())


def comp_traj(x):
    x_obs_mp = x['x_obs_mp'].mean()
    x_obs_ep = x['x_obs_ep'].mean()
    y = x['y'].mean()
    yff = x['yff'].mean()
    xff = x['xff'].mean()
    xff2 = x['xff2'].mean()
    p = np.vstack(x['params'].to_numpy())
    grp = x['group'].unique()[0]
    model = x['model'].unique()[0]
    r_squared_mp = np.round(x['r_squared_mp'].mean(), 2)
    r_squared_ep = np.round(x['r_squared_ep'].mean(), 2)
    subject = x['subject'].to_numpy()[0]
    sig_mp = x['sig_mp'].mean()
    sig_ep = x['sig_ep'].mean()

    ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp))**2)
    ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp))**2)
    ss_res_mp = np.nansum((x_obs_mp - yff)**2)
    ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep))**2)
    ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep))**2)
    ss_res_ep = np.nansum((x_obs_ep - y)**2)

    r_squared_mp_mean = 1 - ss_res_mp / ss_tot_mp
    r_squared_ep_mean = 1 - ss_res_ep / ss_tot_ep
    r_squared_mean = 1 - (ss_res_ep + ss_res_mp) / (ss_tot_ep + ss_tot_mp)

    r_squared_mp_mean = np.round(r_squared_mp_mean, 2)
    r_squared_ep_mean = np.round(r_squared_ep_mean, 2)
    r_squared_mean = np.round(r_squared_mean, 2)

    return {
        'x_obs_mp': x_obs_mp,
        'x_obs_ep': x_obs_ep,
        'y': y,
        'yff': yff,
        'xff': xff,
        'xff2': xff2,
        'p': p,
        'grp': grp,
        'model': model,
        'r_squared_mp': r_squared_mp,
        'r_squared_ep': r_squared_ep,
        'subject': subject,
        'sig_mp': sig_mp,
        'sig_ep': sig_ep,
        'r_squared_mp_mean': r_squared_mp_mean,
        'r_squared_ep_mean': r_squared_ep_mean,
        'r_squared_mean': r_squared_mean,
    }


def fig_grp_scatter(x):

    x = comp_traj(x)

    x_obs_mp = x['x_obs_mp']
    x_obs_ep = x['x_obs_ep']
    y = x['y']
    yff = x['yff']
    xff = x['xff']
    xff2 = x['xff2']
    p = x['p']
    grp = x['grp']
    model = x['model']
    r_squared_mp = x['r_squared_mp']
    r_squared_ep = x['r_squared_ep']
    subject = x['subject']
    sig_mp = x['sig_mp']
    sig_ep = x['sig_ep']
    r_squared_mp_mean = x['r_squared_mp_mean']
    r_squared_ep_mean = x['r_squared_ep_mean']
    r_squared_mean = x['r_squared_mean']

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(14, 6))
    x = np.arange(0, x_obs_mp.shape[0], 1)
    x = x[1:]
    xmp = x_obs_mp[1:]
    sep = sig_ep[:-1]
    smp = sig_mp[:-1]
    ax[0, 0].plot(x, xmp, 'k', alpha=0.2)

    if grp == 15 or grp == 16:
        ax[0, 0].scatter(x[(smp == 1) & (sep == 1)],
                         xmp[(smp == 1) & (sep == 1)],
                         c='C0',
                         label='1')
        ax[0, 0].scatter(x[(smp == 1) & (sep == 3)],
                         xmp[(smp == 1) & (sep == 3)],
                         c='C1',
                         label='2')
        ax[0, 0].scatter(x[(smp == 3) & (sep == 1)],
                         xmp[(smp == 3) & (sep == 1)],
                         c='C2',
                         label='3')
        ax[0, 0].scatter(x[(smp == 3) & (sep == 3)],
                         xmp[(smp == 3) & (sep == 3)],
                         c='C3',
                         label='4')

    else:
        ax[0, 0].scatter(x[sep == 1], xmp[sep == 1], c='C0', label='1')
        ax[0, 0].scatter(x[sep == 2], xmp[sep == 2], c='C1', label='2')
        ax[0, 0].scatter(x[sep == 3], xmp[sep == 3], c='C2', label='3')
        ax[0, 0].scatter(x[sep == 4], xmp[sep == 4], c='C3', label='4')

    plt.legend()
    plt.savefig('../figures/fit_summary_grp_' + str(grp) + '_scatter.pdf')
    plt.close()


def fig_summary_1(x):

    x = comp_traj(x)

    x_obs_mp = x['x_obs_mp']
    x_obs_ep = x['x_obs_ep']
    y = x['y']
    yff = x['yff']
    xff = x['xff']
    xff2 = x['xff2']
    p = x['p']
    grp = x['grp']
    model = x['model']
    r_squared_mp = x['r_squared_mp']
    r_squared_ep = x['r_squared_ep']
    subject = x['subject']
    sig_mp = x['sig_mp']
    sig_ep = x['sig_ep']
    r_squared_mp_mean = x['r_squared_mp_mean']
    r_squared_ep_mean = x['r_squared_ep_mean']
    r_squared_mean = x['r_squared_mean']

    p_names = [
        'alpha_ff_1', 'beta_ff_1', 'bias_ff_1', 'alpha_ff_2', 'beta_ff_2',
        'bias_ff_2', 'alpha_fb', 'beta_fb', 'fb_init', 'gamma_fb_1',
        'gamma_fb_2', 'gamma_fb_3', 'gamma_fb_4', 'gamma_ff_1', 'gamma_ff_2',
        'gamma_ff_3', 'gamma_ff_4', 'temporal_discount'
    ]
    dfp = pd.DataFrame(data=p[:, :-1])
    dfp.columns = p_names
    dfp = dfp[[
        'alpha_ff_1', 'alpha_ff_2', 'beta_ff_1', 'beta_ff_2', 'bias_ff_1',
        'bias_ff_2', 'alpha_fb', 'beta_fb', 'fb_init', 'gamma_fb_1',
        'gamma_fb_2', 'gamma_fb_3', 'gamma_fb_4', 'gamma_ff_1', 'gamma_ff_2',
        'gamma_ff_3', 'gamma_ff_4', 'temporal_discount'
    ]]

    # NOTE: Rescale parameters to (0, 1)
    # dfp['bias_ff_1'] = dfp['bias_ff_1'] / 10
    # dfp['bias_ff_2'] = dfp['bias_ff_2'] / 10
    # dfp['gamma_fb_4'] = dfp['gamma_fb_4'] / 1

    dfp = dfp.drop(columns=['alpha_fb', 'beta_fb', 'fb_init'])
    dfp = dfp.melt()
    dfp['subject'] = dfp.groupby(
        ['variable']).transform(lambda x: np.arange(0, x.shape[0], 1))
    dfp['var_color'] = 'C0'

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 8)

    ax1 = fig.add_subplot(gs[0, :4])
    ax2 = fig.add_subplot(gs[0, 4:])
    x = np.arange(0, x_obs_mp.shape[0], 1)
    ax1.plot(x_obs_mp, label='Human')
    ax1.plot(yff, label='Model Full Output')
    ax1.plot(xff, label='Model State slow')
    ax1.plot(xff2, label='Model State fast')
    ax2.plot(x_obs_ep)
    ax2.plot(y)

    if grp == 15 or grp == 16:
        ax3 = fig.add_subplot(gs[1, :4])
        d = dfp.loc[np.isin(dfp['variable'], ['gamma_ff_1', 'gamma_ff_3'])]
    else:
        ax3 = fig.add_subplot(gs[1, :4])
        d = dfp.loc[np.isin(dfp['variable'],
                            ['gamma_ff_1', 'gamma_ff_2', 'gamma_ff_3'])]

    pg.plot_paired(data=d,
                   dv='value',
                   within='variable',
                   subject='subject',
                   boxplot_in_front=True,
                   ax=ax3)

    if grp == 15 or grp == 16:
        ax4 = fig.add_subplot(gs[1, 4:])
        d = dfp.loc[np.isin(dfp['variable'], ['gamma_fb_1', 'gamma_fb_3'])]
    else:
        ax4 = fig.add_subplot(gs[1, 4:])
        d = dfp.loc[np.isin(dfp['variable'],
                            ['gamma_fb_1', 'gamma_fb_2', 'gamma_fb_3'])]

    pg.plot_paired(data=d,
                   dv='value',
                   within='variable',
                   subject='subject',
                   boxplot_in_front=True,
                   ax=ax4)

    ax5 = fig.add_subplot(gs[2, :2])
    d = dfp.loc[np.isin(dfp['variable'], ['alpha_ff_1', 'alpha_ff_2'])]
    pg.plot_paired(data=d,
                   dv='value',
                   within='variable',
                   subject='subject',
                   boxplot_in_front=True,
                   ax=ax5)
    ax6 = fig.add_subplot(gs[2, 2:4])
    d = dfp.loc[np.isin(dfp['variable'], ['beta_ff_1', 'beta_ff_2'])]
    pg.plot_paired(data=d,
                   dv='value',
                   within='variable',
                   subject='subject',
                   boxplot_in_front=True,
                   ax=ax6)

    ax7 = fig.add_subplot(gs[2, 7])
    d = dfp.loc[np.isin(dfp['variable'], ['bias_ff_2'])]
    pg.plot_paired(data=d,
                   dv='value',
                   within='variable',
                   subject='subject',
                   boxplot_in_front=True,
                   pointplot_kwargs={'scale': 0.0},
                   colors=['black', 'black', 'black'],
                   ax=ax7)

    ax8 = fig.add_subplot(gs[2, 6])
    d = dfp.loc[np.isin(dfp['variable'], ['temporal_discount'])]
    pg.plot_paired(data=d,
                   dv='value',
                   within='variable',
                   subject='subject',
                   boxplot_in_front=True,
                   ax=ax8)

    ax33 = fig.add_subplot(gs[2, 4])
    d33 = dfp.loc[np.isin(dfp['variable'], ['gamma_ff_4'])]
    pg.plot_paired(data=d33,
                   dv='value',
                   within='variable',
                   subject='subject',
                   boxplot_in_front=True,
                   ax=ax33)

    ax44 = fig.add_subplot(gs[2, 5])
    d44 = dfp.loc[np.isin(dfp['variable'], ['gamma_fb_4'])]
    pg.plot_paired(data=d44,
                   dv='value',
                   within='variable',
                   subject='subject',
                   boxplot_in_front=True,
                   ax=ax44)

    ax1.legend()
    ax1.set_title('group ' + str(grp) + ', model ' + str(model) + '\n' +
                  str(r_squared_mp_mean))
    ax2.set_title('group ' + str(grp) + ', model ' + str(model) + '\n' +
                  str(r_squared_ep_mean))
    plt.tight_layout()
    plt.savefig('../figures/fit_summary_grp_' + str(grp) + '_mod_' +
                str(model) + '_subject_' + str(subject) + '.pdf')
    plt.close()


def fig_summary_2(x):
    pass


def plot_bic(x):
    grp = x['group'].unique()
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    pg.plot_paired(data=x,
                   dv='bic',
                   within='model',
                   subject='subject',
                   boxplot_in_front=True,
                   ax=ax)
    plt.savefig('../figures/fit_summary_bic_grp_' + str(grp) + '.pdf')
    plt.close()
    tres = pg.pairwise_ttests(data=x,
                              dv='bic',
                              within='model',
                              subject='subject',
                              padjust='bonf',
                              effsize='cohen',
                              return_desc=True)
    print(tres[['A', 'B', 'T', 'dof', 'p-corr', 'cohen']])
    print(tres['mean(A)'] - tres['mean(B)'])


def report_fit_summary(models, d):

    d = prepare_fit_summary(models, d)

    # TODO: Make new figure
    for grp in d.group.unique():
        d_grp = d.loc[d['group'] == grp]
        fig, ax = plt.subplots(3, 1, squeeze=False)
        for mod in d_grp.model.unique():
            d_mod = d_grp.loc[d_grp['model'] == mod]
            x_obs_mp = d_mod.x_obs_mp.mean()
            sig_mp = d_mod.sig_mp.mean()
            yff = d_mod.yff.mean()
            t = np.arange(1, yff.shape[0] + 1)
            ax[0, 0].plot(t, x_obs_mp, 'k', alpha=0.1)
            ax[1, 0].plot(t, x_obs_mp, 'k', alpha=0.1)
            ax[2, 0].plot(t, x_obs_mp, 'k', alpha=0.1)
            ax[0, 0].scatter(t, x_obs_mp, c='k', alpha=0.5)
            ax[1, 0].scatter(t, x_obs_mp, c=sig_mp, alpha=0.5)
            ax[2, 0].scatter(t, yff, label=mod, alpha=0.5)
        plt.legend()
        plt.show()


    d['pbic'] = d.groupby(['group', 'subject'])['bic'].transform(compute_pbic)

    d['pbic_max'] = d.groupby(
        ['group', 'subject'])['pbic'].transform(lambda x: np.max(x.to_numpy()))

    fit_summary = d.loc[d['pbic'] == d['pbic_max']].groupby(['group', 'model'])

    print()
    print(fit_summary['pbic', 'r_squared', 'r_squared_mp',
                      'r_squared_ep'].mean())
    print()
    print(fit_summary['model'].count())

    d['bic_min'] = d.groupby(
        ['group', 'subject'])['bic'].transform(lambda x: np.min(x.to_numpy()))

    fit_summary = d.loc[d['bic'] == d['bic_min']].groupby(['group', 'model'])

    print()
    print(fit_summary['bic', 'r_squared', 'r_squared_mp',
                      'r_squared_ep'].mean())
    print()
    print(fit_summary['model'].count())


def get_slopes(x):
    group = x['group'].iloc[0]
    subject = x['subject'].iloc[0]
    sig_ep = x['sig_ep'].to_numpy()
    sig_mp = x['sig_mp'].to_numpy()

    rot = x['rot'].to_numpy()
    ep = x['ha_end'].to_numpy()
    mp = x['ha_mid'].to_numpy()

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

            x = error_mp[(sig_mp == smp) & (sig_ep == sep)][:, None]
            y = fb_int[(sig_mp == smp) & (sig_ep == sep)]
            if y.size > 0:
                regr = linear_model.linearregression().fit(x, y)
                slope = regr.coef_[0]
                inter = regr.intercept_
                d['group'].append(group)
                d['subject'].append(subject)
                d['sig_ep'].append(sep)
                d['sig_mp'].append(smp)
                d['slope'].append(slope)
                d['intercept'].append(inter)
                d['process'].append('fb')

            x = error_ep[(sig_mp == smp) & (sig_ep == sep)][:, None]
            y = delta_mp[(sig_mp == smp) & (sig_ep == sep)]
            if y.size > 0:
                regr = linear_model.linearregression().fit(x, y)
                slope = regr.coef_[0]
                inter = regr.intercept_
                d['group'].append(group)
                d['subject'].append(subject)
                d['sig_ep'].append(sep)
                d['sig_mp'].append(smp)
                d['slope'].append(slope)
                d['intercept'].append(inter)
                d['process'].append('ffep')

            x = error_mp[(sig_mp == smp) & (sig_ep == sep)][:, None]
            y = delta_mp[(sig_mp == smp) & (sig_ep == sep)]
            if y.size > 0:
                regr = linear_model.linearregression().fit(x, y)
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

    ncols = d['group'].unique().size
    nrows = 3

    fig, ax = plt.subplots(nrows, ncols, squeeze=False)

    cc = np.array(['c0', 'c1', 'c2', 'c3', 'c4'])

    for g, grp in enumerate(np.sort(d['group'].unique())):

        d_grp = d.loc[d['group'] == grp]

        rot = np.vstack(d_grp['rot'].to_numpy()[:]).flatten()
        ep = np.vstack(d_grp['ha_end'].to_numpy()[:]).flatten()
        mp = np.vstack(d_grp['ha_mid'].to_numpy()[:]).flatten()
        # mp = np.vstack(d_grp['ha_init'].to_numpy()[:]).flatten()
        delta_mp = np.diff(mp, append=0)
        fb_int = ep - mp
        sig_mp = np.vstack(d_grp['sig_mp'].to_numpy()[:]).flatten()
        sig_ep = np.vstack(d_grp['sig_ep'].to_numpy()[:]).flatten()
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
                    ax[0, g].set_xlabel('mp error')
                    ax[0, g].set_ylabel('fb integration')
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
                    ax[1, g].set_xlabel('ep error')
                    ax[1, g].set_ylabel('ff adaptation')
                    ax[1, g].set_xticks([])
                    ax[1, g].set_yticks([])
                    ax[1, g].set_title('group ' + str(grp))
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
                    ax[2, g].set_xlabel('mp error')
                    ax[2, g].set_ylabel('ff adaptation')
                    # ax[2, g].set_xticks([])
                    # ax[2, g].set_yticks([])
                    # ax[2, g].set_title('group ' + str(grp))
                    ax[2, g].legend(loc='lower left')

    # plt.tight_layout()
    plt.show()


def report_parameter_estimates(models, d):

    dfs = prepare_fit_summary(models, d)
    pname = np.array([
        'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'xfb_init', 'w',
        'gamma_ff_1', 'gamma_ff_2', 'gamma_ff_3', 'gamma_fbint_1',
        'gamma_fbint_2', 'gamma_fbint_3', 'sse'
    ])
    n_params = pname.shape[0] - 1
    nrows = 2
    ncols = 2
    # ncols = dfs['group'].unique().shape[0]

    fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    ax = np.reshape(ax, (1, 4))
    for g, grp in enumerate(dfs['group'].unique()):

        params = dfs.loc[dfs['model'] == 'all', ['group', 'subject', 'params']]
        params = np.vstack(params.loc[(params['group'] == grp),
                                      'params'].to_numpy())

        x = np.arange(1, params.shape[1], 1)
        ax[0, g].plot([1, n_params], [0, 0], '--')
        ax[0, g].violinplot(params[:, :-1])
        ax[0, g].set_xticks(x)
        ax[0, g].set_xticklabels(pname[:-1])
        ax[0, g].set_title(str(grp))
        for jj in range(params.shape[0]):
            ax[0, g].plot(x, params[jj, :-1], '.', alpha=0.5)

        tstat, pval = ttest_1samp(params, popmean=0, axis=0)
        cd = np.mean(params, axis=0) / np.std(params, axis=0, ddof=1)

        print('\n')
        print('group ' + str(grp))
        inds = np.arange(0, 11, 1)
        for j in inds:
            print(pname[j] + ' = ' + str(np.round(params[:, j].mean(), 2)) +
                  ': t(' + str(params.shape[0] - 1) + ') = ' +
                  str(np.round(tstat[j], 2)) + ', p = ' +
                  str(np.round(pval[j], 2)) + ', d = ' +
                  str(np.round(cd[j], 2)))

    plt.tight_layout()
    plt.show()

    # # todo: for each best fit model, plot mean observed vs mean predicted
    # for g, grp in enumerate(dfs['group'].unique()):
    #     d_grp = dfs.loc[dfs['group'] == grp]
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

    #         ax[0, s].plot(mp, '.c0')
    #         ax[0, s].plot(yff, '.c1')
    #         ax[1, s].plot(ep, '.c0')
    #         ax[1, s].plot(y, '.c1')
    #     plt.show()

    # s2 = dfs.loc[dfs['model'] == dfs['best_model'],
    #               ['group', 'best_model', 'x_obs_ep', 'x_obs_mp', 'y', 'yff']]

    # s2 = s2.melt(id_vars=['group', 'best_model'],
    #              value_vars=['x_obs_ep', 'x_obs_mp', 'y', 'yff'])

    # s3 = s2.groupby(['group', 'best_model', 'variable'
    #                  ]).apply(lambda x: np.vstack(x.to_numpy()).mean(0)[0])

    # s3 = s3.to_frame('to_numpy()').reset_index()

    # s4 = s3.pivot(index=none, columns='variable', to_numpy()='to_numpy()')

    # n_groups = s3['group'].unique().shape[0]
    # n_best_models = s3['best_model'].unique().shape[0]
    # fig, ax = plt.subplots(n_groups, n_best_models)
    # for g, grp, in enumerate(s3['group'].unique()):
    #     for m, model in enumerate(s3['best_model'].unique()):
    #         ep = s3.loc[(s3['group'] == grp) & (s3['best_model'] == model) &
    #                     (s3['variable'] == 'x_obs_ep'), 'to_numpy()'].to_numpy()
    #         mp = s3.loc[(s3['group'] == grp) & (s3['best_model'] == model) &
    #                     (s3['variable'] == 'x_obs_mp'), 'to_numpy()'].to_numpy()
    #         if ep.size > 0:
    #             ep = ep[0]
    #             mp = mp[0]

    #         ax[g, m].plot(ep, '.c0')
    #         ax[g, m].plot(mp, '.c1')
    #         ax[g, m].get_xaxis().set_ticks([])
    #         ax[g, m].get_yaxis().set_ticks([])
    #         ax[g, m].set_title(grp)
    # plt.show()

    # todo: for each best fit model, validate the fit parameters and selection


def inspect_fits_boot(group):

    d = load_all_data()


def fit_individual(modname, d, fit_args, froot):

    obj_func = fit_args['obj_func']
    bounds = fit_args['bounds']
    constraints = fit_args['constraints']
    maxiter = fit_args['maxiter']
    disp = fit_args['disp']
    tol = fit_args['tol']
    polish = fit_args['polish']
    updating = fit_args['updating']
    workers = fit_args['workers']
    popsize = fit_args['popsize']
    mutation = fit_args['mutation']
    recombination = fit_args['recombination']

    for grp in d['group'].unique():
        for sub in d[d['group'] == grp]['subject'].unique():

            dd = d[(d['subject'] == sub) & (d['group'] == grp)][[
                'rot', 'ha_init', 'ha_end', 'trial_abs', 'group', 'sig_mp',
                'sig_ep'
            ]]

            rot = dd.rot.to_numpy()
            sig_mp = dd.sig_mp.to_numpy()
            sig_ep = dd.sig_ep.to_numpy()
            group = dd.group.to_numpy()
            x_obs_mp = dd['ha_init'].to_numpy()
            x_obs_ep = dd['ha_end'].to_numpy()

            args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep, group, modname)

            results = differential_evolution(func=obj_func,
                                             bounds=bounds,
                                             constraints=constraints,
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

            fout = froot + '_group_' + str(grp) + '_sub_' + str(sub) + '.txt'
            with open(fout, 'w') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def fit_boot(credit, d, fit_args, froot):

    n_boot_samp = 0

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

    for grp in d['group'].unique():

        dd = d[d['group'] == grp]

        ddd = dd.groupby('trial_abs').mean().reset_index()
        ddd = ddd.sort_values('trial_abs')

        rot = ddd.rot.to_numpy()
        sig_mp = ddd.sig_mp.to_numpy().astype(int)
        sig_ep = ddd.sig_ep.to_numpy().astype(int)
        group = ddd.group.to_numpy().astype(int)
        x_obs_mp = ddd['ha_init'].to_numpy()
        # x_obs_mp = ddd['ha_mid'].to_numpy()
        x_obs_ep = ddd['ha_end'].to_numpy()

        args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep, group, credit)

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

        fout = froot + '_group_' + str(grp) + '_grp.txt'
        with open(fout, 'w') as f:
            tmp = np.concatenate((results['x'], [results['fun']]))
            tmp = np.reshape(tmp, (tmp.shape[0], 1))
            np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')

        for n in range(n_boot_samp):

            boot_subs = np.random.choice(d['subject'].unique(),
                                         size=d['subject'].unique().shape[0],
                                         replace=True)

            ddd = []
            for i in range(boot_subs.shape[0]):

                ddd.append(dd[dd['subject'] == boot_subs[i]][[
                    'rot', 'ha_init', 'ha_end', 'trial_abs', 'group', 'sig_mp',
                    'sig_ep'
                ]])

            ddd = pd.concat(ddd)
            ddd = ddd.groupby('trial_abs').mean().reset_index()
            ddd = ddd.sort_values('trial_abs')

            rot = ddd.rot.to_numpy()
            sig_mp = ddd.sig_mp.to_numpy().astype(int)
            sig_ep = ddd.sig_ep.to_numpy().astype(int)
            group = ddd.group.to_numpy().astype(int)
            x_obs_mp = ddd['ha_init'].to_numpy()
            x_obs_ep = ddd['ha_end'].to_numpy()

            args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep, group, credit)

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

            fout = froot + '_group_' + str(grp) + '_boot.txt'
            with open(fout, 'a') as f:
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
    modname = obs[6]

    args = (rot, sig_mp, sig_ep, group, modname)

    x_pred = simulate(params, args)
    x_pred_mp = x_pred[1]
    x_pred_ep = x_pred[0]

    sse_mp = np.sum((x_obs_mp - x_pred_mp)**2)
    sse_ep = np.sum((x_obs_ep - x_pred_ep)**2)

    sse = sse_mp + sse_ep

    return sse


def obj_func_nll(params, *args):

    obs = args

    rot = obs[0]
    sig_mp = obs[1]
    sig_ep = obs[2]
    x_obs_mp = obs[3]
    x_obs_ep = obs[4]
    group = obs[5]
    modname = obs[6]

    args = (rot, sig_mp, sig_ep, group, modname)

    x_pred = simulate(params, args)
    x_pred_mp = x_pred[1]
    x_pred_ep = x_pred[0]

    # how likely are your data given the current parameters
    l = norm.pdf(x_obs_mp, loc=x_pred_mp, scale=params[-1])
    nll = -np.log(l)
    nll_mp = np.sum(nll)

    l = norm.pdf(x_obs_ep, loc=x_pred_ep, scale=params[-1])
    nll = -np.log(l)
    nll_ep = np.sum(nll)

    nll = nll_mp + nll_ep

    return nll


def simulate(params, args):

    alpha_ff = params[0]
    beta_ff = params[1]
    bias_ff = params[2]
    alpha_ff2 = params[3]
    beta_ff2 = params[4]
    bias_ff2 = params[5]
    alpha_fb = params[6]
    beta_fb = params[7]
    xfb_init = params[8]

    gamma_fbint_1 = params[9]
    gamma_fbint_2 = params[10]
    gamma_fbint_3 = params[11]
    gamma_fbint_4 = params[12]
    gamma_fbint = np.array(
        [gamma_fbint_1, gamma_fbint_2, gamma_fbint_3, gamma_fbint_4])

    gamma_ff_1 = params[13]
    gamma_ff_2 = params[14]
    gamma_ff_3 = params[15]
    gamma_ff_4 = params[16]
    gamma_ff = np.array([gamma_ff_1, gamma_ff_2, gamma_ff_3, gamma_ff_4])

    td = params[17]

    r = args[0]
    sig_mp = args[1]
    sig_ep = args[2]
    group = args[3]
    modname = args[4]

    n_trials = r.shape[0]

    delta_ep = np.zeros(n_trials)
    delta_mp = np.zeros(n_trials)
    xff = np.zeros(n_trials)
    xfb = np.zeros(n_trials)
    yff = np.zeros(n_trials)
    yfb = np.zeros(n_trials)
    y = np.zeros(n_trials)

    xff2 = np.zeros(n_trials)

    xfb[0] = xfb_init

    for i in range(n_trials - 1):

        ff_adapt_mp = 0.0
        ff_adapt_ep = 0.0
        ff_adapt_mp2 = 0.0
        ff_adapt_ep2 = 0.0

        # start to midpoint
        yff[i] = xff[i] + xff2[i]
        yfb[i] = 0.0
        y[i] = yff[i] + yfb[i]

        if sig_mp[i] != 4:
            delta_mp[i] = 0.0 - (y[i] + r[i])
            yfb[i] = xfb[i] * delta_mp[i] * gamma_fbint[sig_mp[i] - 1]
            ff_adapt_mp = alpha_ff * delta_mp[i]
            ff_adapt_mp2 = alpha_ff2 * delta_mp[i]
        else:
            delta_mp[i] = 0.0
            yfb[i] = gamma_fbint[sig_mp[i] - 1]

        # midpoint to endpoint
        y[i] = yff[i] + yfb[i]

        if sig_ep[i] != 4:
            delta_ep[i] = 0.0 - (y[i] + r[i])
            ff_adapt_ep = alpha_ff * (delta_ep[i] - yfb[i])
            ff_adapt_ep2 = alpha_ff2 * (delta_ep[i] - yfb[i])
        else:
            delta_ep[i] = 0.0

        # update fb state
        xfb[i + 1] = beta_fb * xfb[i] + alpha_fb * delta_ep[i]

        # NOTE: some condition files call a 4 a 0?
        if sig_ep[i] == 0:
            sig_ep[i] = 4

        # update ff state ( all other groups )
        xff[i + 1] = beta_ff * xff[i] + ff_adapt_mp + ff_adapt_ep + bias_ff

        if modname == 'error-scale' or modname == 'error-scale-one-state':
            xff2[i + 1] = beta_ff2 * xff2[i] + td * gamma_ff[
                sig_mp[i] - 1] * ff_adapt_mp2 + (
                    1 - td) * gamma_ff[sig_ep[i] - 1] * ff_adapt_ep2 + bias_ff2

        elif modname == 'state-scale' or modname == 'state-scale-one-state':
            xff2[i +
                 1] = (td * gamma_ff[sig_mp[i] - 1] +
                       (1 - td) * gamma_ff[sig_ep[i] - 1]) * beta_ff2 * xff2[
                           i] + ff_adapt_mp2 + ff_adapt_ep2 + bias_ff2

        elif modname == 'bias-scale' or modname == 'bias-scale-one-state':
            xff2[i + 1] = beta_ff2 * xff2[i] + ff_adapt_mp2 + ff_adapt_ep2 + (
                td * gamma_ff[sig_mp[i] - 1] +
                (1 - td) * gamma_ff[sig_ep[i] - 1]) * bias_ff2

        # clip the feedback gain to prevent instability
        xfb = np.clip(xfb, -2, 2)

    return (y, yff, yfb, xff, xfb, xff2)


def bayes_int(x, m):

    # x = np.arange(-1, 3, 0.01)
    # plt.plot(x, np.tanh(0 * (x - 2)) / 2 + 0.5, label=str(0))
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
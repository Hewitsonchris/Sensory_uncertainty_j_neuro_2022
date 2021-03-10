from imports import *
from util_funcs import *

d = load_all_data()
# inspect_behaviour_all(d)

grp = [3, 4, 5]
# grp = [0]
# grp = [14]
# grp = [13]
# grp = [3, 4, 5, 0, 13, 14]
# grp = [14]

dd = d.loc[np.isin(d['group'], grp)]
dd = dd.loc[np.isin(dd['phase'], ['adaptation'])]
dd = dd[dd['trial'] <= 180]
dd = dd.reset_index(drop=True)

dd.loc[dd['group'] == 3, 'group'] = 345
dd.loc[dd['group'] == 4, 'group'] = 345
dd.loc[dd['group'] == 5, 'group'] = 345

dd = dd.loc[dd['subject'] == 1]

# dd.loc[dd['group'] == 345, 'ha_init'] = dd.loc[dd['group'] == 345, 'ha_init'] * 10
# dd.loc[dd['group'] == 345, 'ha_mid'] = dd.loc[dd['group'] == 345, 'ha_mid'] * 10
# dd.loc[dd['group'] == 345, 'ha_end'] = dd.loc[dd['group'] == 345, 'ha_end'] * 10 # dd.loc[dd['group'] == 345, 'rot'] = dd.loc[dd['group'] == 345, 'rot'] * 10

# dd.loc[dd['group'] == 14, 'ha_init'] = dd.loc[dd['group'] == 14, 'ha_init'] * 10
# dd.loc[dd['group'] == 14, 'ha_mid'] = dd.loc[dd['group'] == 14, 'ha_mid'] * 10
# dd.loc[dd['group'] == 14, 'ha_end'] = dd.loc[dd['group'] == 14, 'ha_end'] * 10
# dd.loc[dd['group'] == 14, 'rot'] = dd.loc[dd['group'] == 14, 'rot'] * 10


# Check trial order
# def check_sequence(x):
#     for s in x['subject'].unique():
#         ds = x.loc[x['subject'] == s]
#         print(ds[['trial', 'trial_abs', 'sig_mp']])


# dd.loc[dd['group'] == 345].groupby(['group']).apply(check_sequence)

models = define_models()
fit_models(models, dd)
report_fit_summary(models, dd)
# report_parameter_estimates(models, dd)
# inspect_behaviour(dd)
# inspect_behaviour_2(models, dd)
# report_fit_summary_boot(models, dd)

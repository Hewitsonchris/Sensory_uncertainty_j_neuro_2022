from imports import *
from util_funcs_matt import *

d = load_all_data()

# 2022 Paper 1 group numbers
# group 15: unimodal stochastic (n=20, low uncertainty, high uncertainty [2x2], exp1 2022 paper 1)
# group 19: unimodal stochastic (n=20, low uncertainty, medium uncertainty, high uncertainty, no-fb, 'Midpoint only' exp2 2022 paper 1)
# group 20: unimodal stochastic (n=20, low uncertainty, medium uncertainty, high uncertainty, no-fb, 'Midpoint plus enpoint incongruent' exp3 2022 paper 1)

# grp = [0, 14, 345, 17,  18,  19,  20]
# grp = [15, 16, 17, 18, 19, 20]


dd = d.loc[np.isin(d['group'], grp)]
dd = dd.loc[np.isin(dd['phase'], ['adaptation', 'washout'])]
dd = dd.reset_index(drop=True)

dd.loc[dd['group'] == 3, 'group'] = 345
dd.loc[dd['group'] == 4, 'group'] = 345
dd.loc[dd['group'] == 5, 'group'] = 345


models = define_models()
#fit_models(models, dd)
report_fit_summary(models, dd)
report_parameter_estimates(models, dd)
#inspect_behaviour(dd)
# inspect_behaviour_2(models, dd)
# report_fit_summary_boot(models, dd)

from imports import *
from util_funcs import *

d = load_all_data()
# inspect_behaviour_all(d)

# grp = [0, 3, 4, 5]
grp = [0, 13, 14, 3, 4, 5]
dd = d.loc[np.isin(d['group'], grp)]
dd = dd.loc[np.isin(dd['phase'], ['adaptation'])]
dd = dd[dd['trial'] <= 180]
dd = dd.reset_index(drop=True)
dd.loc[dd['group'] == 3, 'group'] = 345
dd.loc[dd['group'] == 4, 'group'] = 345
dd.loc[dd['group'] == 5, 'group'] = 345

models = define_models()
# fit_models(models, dd)
report_fit_summary(models, dd)
report_parameter_estimates(models, dd)
# inspect_behaviour(dd)
# inspect_behaviour_2(models, dd)

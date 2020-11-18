from imports import *
from util_funcs import *

d = load_all_data()
# inspect_behaviour_all(d)

grp = [0, 13, 14, 345]
dd = d.set_index('group', drop=False).loc[grp]
dd = dd.set_index('phase', drop=False).loc['adaptation']
dd = dd[dd['trial'] <= 180]
dd = dd.reset_index(drop=True)
dd.loc[dd['group'] == 3, 'group'] = 345
dd.loc[dd['group'] == 4, 'group'] = 345
dd.loc[dd['group'] == 5, 'group'] = 345

# fit_models(dd)
# inspect_fits_individual_model_compare(dd)
# inspect_behaviour(dd)

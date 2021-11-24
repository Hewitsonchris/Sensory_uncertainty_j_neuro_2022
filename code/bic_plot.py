import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

d = pd.read_csv('../data/BIC_long.csv')
d = d.loc[d['GROUP'] == 15]

ax = sns.violinplot(data=d, x='MODEL', y='BIC', inner='stick')

pairs = [('bias_two', 'bias_single'), ('bias_two', 'error_single'),
         ('bias_two', 'error_two'), ('bias_two', 'state_single'),
         ('bias_two', 'state_two')]

order = [
    'bias_single', 'bias_two', 'error_single', 'error_two', 'state_single',
    'state_two'
]

annotator = Annotator(ax, pairs, data=d, x='MODEL', y='BIC', order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside')
annotator.apply_and_annotate()

plt.show()

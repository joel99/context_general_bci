#%%
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from context_general_bci.analyze_utils import prep_plt
split = 'm1'
split = 'm2'
eval_path = Path(f"data/falcon_continual/{split}_eval_ndt2.csv")

df = pd.read_csv(eval_path)
if split == 'm2':
    df = df[df['variant'] == 'm2_chop_2s-sweep-chop']
    print(df)

palette = sns.color_palette(n_colors=2)
ax = prep_plt()
ax = sns.lineplot(data=df, x='augment_chop_length_ms', y='eval_r2', ax=ax)
ax = sns.lineplot(data=df, x='augment_chop_length_ms', y='heldin_eval_r2', ax=ax)
low_hi_trialized = {
    'm1': (0.557, 0.761),
    'm2': (0.387, 0.586),
}
ax.axhline(low_hi_trialized[split][0], color=palette[0], linestyle='--', label='Trialized')
ax.axhline(low_hi_trialized[split][1], color=palette[1], linestyle='--')
ax.legend()
ax.set_title(f'{split} NDT2 Trialized vs Continual Eval')
ax.set_ylabel('R2')
ax.set_xlabel('Chop Length (ms)')
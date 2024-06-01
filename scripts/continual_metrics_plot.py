#%%
# Compute oracle FALCON metrics by identifying the best run in a group
import os
# set cuda to device 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl

from falcon_challenge.evaluator import FalconEvaluator, DATASET_HELDINOUT_MAP

pl.seed_everything(0)

# argparse the eval set
import sys
import argparse

num_workers = 4 # for main eval block.
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    VARIANT = 'h1'
    # VARIANT = 'm2'
    # VARIANT = 'm1'
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", "-v", type=str, required=True, choices=[
            'h1', 
            'm1',
            'm2',
        ]
    )
    args = parser.parse_args()
    VARIANT = args.variant

eval_paths = Path('./data/falcon_metrics')
eval_paths.mkdir(exist_ok=True, parents=True)
def get_ndt2_run_df_for_variant(variant):
    eval_metrics_path_cont = eval_paths / f"{variant}_continual_ndt2.csv"
    eval_metrics_path_tr = eval_paths / f"{variant}_trialized_ndt2.csv"
    ndt2_run_df_cont = pd.read_csv(eval_metrics_path_cont) if eval_metrics_path_cont.exists() else pd.DataFrame()
    ndt2_run_df_tr = pd.read_csv(eval_metrics_path_tr) if eval_metrics_path_tr.exists() else pd.DataFrame()
    ndt2_run_df_cont['type'] = 'continual'
    ndt2_run_df_tr['type'] = 'trialized'
    ndt2_run_df = pd.concat([ndt2_run_df_cont, ndt2_run_df_tr])
    ndt2_run_df['variant'] = variant
    return ndt2_run_df

# ndt2_run_df = pd.concat([get_ndt2_run_df_for_variant(variant) for variant in ['h1', 'm1', 'm2']])
ndt2_run_df = pd.concat([get_ndt2_run_df_for_variant(variant) for variant in ['h1', 'm2']])
print(ndt2_run_df.variant.unique())
    
#%%
from context_general_bci.analyze_utils import prep_plt
ax = prep_plt()
ndt2_run_df.head()
ndt2_run_df['perf_diff'] = ndt2_run_df['heldin_eval_r2'] - ndt2_run_df['eval_r2']
ax = sns.lineplot(data=ndt2_run_df, x='augment_chop_length_ms', y='eval_r2', hue='type', ax=ax)
# ax = sns.lineplot(data=ndt2_run_df, x='augment_chop_length_ms', y='eval_r2', hue='type', ax=ax)
# ax = sns.barplot(data=ndt2_run_df, x='augment_chop_length_ms', y='heldin_eval_r2', hue='type', ax=ax)
ax = sns.lineplot(data=ndt2_run_df, x='augment_chop_length_ms', y='heldin_eval_r2', hue='type', ax=ax)
# sns.lineplot(data=ndt2_run_df, x='augment_chop_length_ms', y='eval_r2', hue='type')
ax.set_ylim(0, 1)

ax.set_xlabel('History Length (ms)')
ax.set_ylabel("Behavior $R^2$")

#%%
trialized_model_scores = {
    # computed through separate ndt2_sample runs with the provided sample ckpts
    'h1': {
        'trialized': {
            'eval_r2': 0.546,
            'heldin_eval_r2': 0.728,
            'Held Out R2 Std.': 0.063,
            'Held In R2 Std.': 0.041,
        },
        'continual': {
            'eval_r2': 0.521,
            'heldin_eval_r2': 0.624,
            'Held Out R2 Std.': 0.044,
            'Held In R2 Std.': 0.040,
        }
    },
    'm1': {
        'trialized': {
            'eval_r2': 0.0,
            'heldin_eval_r2': 0.0,
            'Held Out R2 Std.': 0.0,
            'Held In R2 Std.': 0.0,
        },
        'continual': {
            'eval_r2': 0.0,
            'heldin_eval_r2': 0.0,
            'Held Out R2 Std.': 0.0,
            'Held In R2 Std.': 0.0,
        }
    },
    'm2': {
        'trialized': {
            'eval_r2': 0.397,
            'heldin_eval_r2': 0.563,
            'Held Out R2 Std.': 0.075,
            'Held In R2 Std.': 0.005,
        },
        'continual': {
            'eval_r2': 0.056,
            'heldin_eval_r2': 0.08,
            'Held Out R2 Std.': 0.005,
            'Held In R2 Std.': 0.007,
        }
    }
}

palette = sns.color_palette(n_colors=2)
f = plt.figure(figsize=(6,6))
ax = prep_plt(f.gca(), big=True)

def plot_single(ax, split, df):
    df_long = df.melt(
        id_vars=['augment_chop_length_ms', 'type'],
        value_vars=['eval_r2', 'heldin_eval_r2'],
        var_name='in_out',
        value_name='r2_value'
    )
    # print(df_long)
    # df_std_long = df.melt(
    #     id_vars=['augment_chop_length_ms', 'type'],
    #     value_vars=['Held Out R2 Std.', 'Held In R2 Std.'],
    #     var_name='in_out',
    #     value_name='r2_std'
    # ) # Abandon error - seaborn doesn't have a good way of attaching

    # Create the line plots (with linestyles)
    sns.lineplot(
        data=df_long,
        x='augment_chop_length_ms',
        y='r2_value',
        hue='in_out',
        style='type',  # Use metric to control linestyle (solid vs. dashed)
        ax=ax,
        legend=False, # Remove default legend
        palette=palette,
    )
    sns.scatterplot(
        data=df_long,
        x='augment_chop_length_ms',
        y='r2_value',
        hue='in_out',
        # style='metric',  # Use metric to control linestyle (solid vs. dashed)
        ax=ax,
        legend=False, # Remove default legend
        palette=palette
    )

    def plot_base_mean_std(mean, std, label, color, style):
        ax.axhline(y=mean, color=color, linestyle=style, label=label)
        ax.axhspan(
            mean - std,
            mean + std,
            color=color, alpha=0.1
        )
    plot_base_mean_std(
        trialized_model_scores[split]['continual']['eval_r2'],
        trialized_model_scores[split]['continual']['Held Out R2 Std.'],
        'Continual Eval R2',
        palette[0],
        '-.'
    )
    plot_base_mean_std(
        trialized_model_scores[split]['continual']['heldin_eval_r2'],
        trialized_model_scores[split]['continual']['Held In R2 Std.'],
        'Continual Heldin Eval R2',
        palette[1],
        '-.'
    )
    plot_base_mean_std(
        trialized_model_scores[split]['trialized']['eval_r2'],
        trialized_model_scores[split]['trialized']['Held Out R2 Std.'],
        'Trialized Eval R2',
        palette[0],
        ':',
    )
    plot_base_mean_std(
        trialized_model_scores[split]['trialized']['heldin_eval_r2'],
        trialized_model_scores[split]['trialized']['Held In R2 Std.'],
        'Trialized Heldin Eval R2',
        palette[1],
        ':',
    )
    
    # Customize linestyles
    linestyles = {"continual": "-", "trialized": "--"}
    for artist in ax.lines:
        if artist.get_label() in linestyles:
            artist.set_linestyle(linestyles[artist.get_label()])

    # Add text annotations
    types = ndt2_run_df['type'].unique()
    colors = {t: c for t, c in zip(types, sns.color_palette())}
    max_x = ndt2_run_df['augment_chop_length_ms'].max()

    x_coords_to_annotate = df_long[df_long['r2_value'] < 0]['augment_chop_length_ms']
    y_coords_to_annotate = df_long[df_long['r2_value'] < 0]['r2_value']
    print(x_coords_to_annotate, y_coords_to_annotate)
    for x, y in zip(x_coords_to_annotate, y_coords_to_annotate):
        ax.annotate("",
                    xy=(x, y),
                    xytext=(x, y - 0.05),  # Adjust the offset to control arrow length
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

    ax.set_ylim(0, 1)  # Adjust y-axis limits if needed

    # for type in types:
    #     data_for_type = ndt2_run_df_long[(ndt2_run_df_long['type'] == type) & (ndt2_run_df_long['metric'] == 'heldin_eval_r2')]
    #     max_y_for_type = data_for_type['r2_value'].max()
    #     ax.text(max_x, max_y_for_type, type, ha='left', va='center', color=colors[type])
    
    ax.set_ylim(0, 1)
    ax.set_xlabel('History Length (ms)')
    ax.set_ylabel(r"Behavior $R^2$")
    ax.set_title(f'{split.upper()}')
    return ax
plot_single(ax, 'h1', ndt2_run_df[ndt2_run_df['variant'] == 'h1'])
# plot_single(ax, 'm2', ndt2_run_df[ndt2_run_df['variant'] == 'm2'])
# plot_single(ax, VARIANT, ndt2_run_df[ndt2_run_df['variant'] == 'H1'])
# plot_single(VARIANT, ndt2_run_df[ndt2_run_df['variant'] == 'H1'])
# %%

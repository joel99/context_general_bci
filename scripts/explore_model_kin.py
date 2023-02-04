#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
from model import transfer_model, logger

from analyze_utils import stack_batch, get_wandb_run, load_wandb_run, wandb_query_latest
from analyze_utils import prep_plt

query = "indy_base_decode"
query = "pitt_obs_decode"

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, exact=True, allow_running=True)[0]
print(wandb_run.id)

# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='co_bps')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='bps')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
# cfg.dataset.datasets = cfg.dataset.datasets[:1]
# cfg.model.task.tasks = [ModelTask.infill]
cfg.model.task.metrics = [Metric.kinematic_r2]
# cfg.model.task.metrics = [Metric.bps, Metric.all_loss]
cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]
# cfg.dataset.datasets = cfg.dataset.datasets[-1:]
# cfg.dataset.datasets = ['mc_maze$']
# cfg.dataset.datasets = ['mc_maze_large']
# cfg.dataset.datasets = ['mc_maze_medium']
# cfg.dataset.datasets = ['mc_maze_small']
# cfg.dataset.datasets = ['churchland_misc_jenkins-10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI']
# cfg.dataset.datasets = ['churchland_misc_reggie-1413W9XGLJ2gma1CCXpg1DRDGpl4-uxkG']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170215_02']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170214_02']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170213_02']

# cfg.dataset.datasets = ['mc_rtt']
if 'rtt' in query:
    cfg.dataset.datasets = ['odoherty_rtt-Indy-20161005_06']
    # cfg.dataset.datasets = ['odoherty_rtt-Indy-20161014_04']
# cfg.dataset.eval_datasets = []
print(cfg.dataset.datasets)
dataset = SpikingDataset(cfg.dataset)
if cfg.dataset.eval_datasets:
    dataset.subset_split(splits=['eval'])
else:
    dataset.subset_split()
dataset.build_context_index()
data_attrs = dataset.get_data_attrs()
print(data_attrs)
# data_attrs.context.session = ['ExperimentalTask.odoherty_rtt-Indy-20161014_04'] # definitely using..
model = transfer_model(src_model, cfg.model, data_attrs)
print(f'{len(dataset)} examples')
trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
# print(context_registry.query(alias='Mihi'))
# model.cfg.task.outputs = [Output.heldout_logrates]
# model.cfg.task.metrics = [Metric.bps]
def get_dataloader(dataset: SpikingDataset, batch_size=100, num_workers=1, **kwargs) -> DataLoader:
    # Defaults set for evaluation on 1 GPU.
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collater_factory()
    )

dataloader = get_dataloader(dataset)
#%%
print(query)
heldin_metrics = stack_batch(trainer.test(model, dataloader))
heldin_outputs = stack_batch(trainer.predict(model, dataloader))

#%%
# B T H
# The predictions here are definitely not correct...
# So why do we have low error?
# ?
trials = 1
print(heldin_outputs[Output.behavior][0,:10,0])
print(heldin_outputs[Output.behavior_pred][0,:10,0])
print(heldin_outputs[Output.behavior].mean())
print(heldin_outputs[Output.behavior].max())
print(heldin_outputs[Output.behavior].min())
# sns.histplot(heldin_outputs[Output.behavior].flatten())
# sns.histplot(heldin_outputs[Output.behavior_pred].flatten())
ax = prep_plt()
sns.scatterplot(x=heldin_outputs[Output.behavior].flatten(), y=heldin_outputs[Output.behavior_pred].flatten(), ax=ax)
#%%
ax = prep_plt()
trials = 4
colors = sns.color_palette('colorblind', trials)
def plot_trial(trial, ax, color):
    vel_true = heldin_outputs[Output.behavior][trial]
    vel_pred = heldin_outputs[Output.behavior_pred][trial]
    pos_true = vel_true.cumsum(0)
    pos_pred = vel_pred.cumsum(0)
    ax.plot(pos_true[:,0], pos_true[:,1], label='true', linestyle='-', color=color)
    ax.plot(pos_pred[:,0], pos_pred[:,1], label='pred', linestyle='--', color=color)

for i in range(trials):
    plot_trial(i, ax, colors[i])
#%%
# print(heldin_outputs[Output.rates].max(), heldin_outputs[Output.rates].mean())
# test = heldin_outputs[Output.heldout_rates]
rates = heldin_outputs[Output.rates] # b t c


spikes = [rearrange(x, 't a c -> t (a c)') for x in heldin_outputs[Output.spikes]]
ax = prep_plt()

num = 20
# channel = 5
# channel = 10
# channel = 18
# channel = 19
# channel = 20
# channel = 80

colors = sns.color_palette("husl", num)

# for trial in range(num):
#     ax.plot(rates[trial][:,channel], color=colors[trial])

y_lim = ax.get_ylim()[1]
# plot spike raster
# for trial in range(num):
#     spike_times = spikes[trial,:,channel].nonzero()
#     y_height = y_lim * (trial+1) / num
#     ax.scatter(spike_times, torch.ones_like(spike_times)*y_height, color=colors[trial], s=10, marker='|')

trial = 10
trial = 15
# trial = 17
# trial = 18
# trial = 80
# trial = 85
for channel in range(num):
    # ax.scatter(np.arange(test.shape[1]), test[0,:,channel], color=colors[channel], s=1)
    ax.plot(rates[trial][:,channel * 2], color=colors[channel])
    # ax.plot(rates[trial][:,channel * 3], color=colors[channel])

    # smooth the signal with a gaussian kernel

# from scipy import signal
# peaks, _ = signal.find_peaks(test[trial,:,2], distance=4)
# print(peaks)
# print(len(peaks))
# for p in peaks:
#     ax.axvline(p, color='k', linestyle='--')



ax.set_ylabel('FR (Hz)')
ax.set_yticklabels((ax.get_yticks() * 1000 / cfg.dataset.bin_size_ms).round())
# relabel xtick unit from 5ms to ms
ax.set_xlim(0, 50)
ax.set_xticklabels(ax.get_xticks() * cfg.dataset.bin_size_ms)
ax.set_xlabel('Time (ms)')
# plt.plot(test[0,:,0])
ax.set_title(f'FR Inference: {query}')

#%%
# Debugging (for mc_maze dataset)
pl.seed_everything(0)
example_batch = next(iter(dataloader))
print(example_batch[DataKey.spikes].size())
print(example_batch[DataKey.spikes].sum())
# print(example_batch[DataKey.spikes][0,:,0,:,0].nonzero())
# First 10 timesteps, channel 8 fires 3x
print(example_batch[DataKey.spikes][0,:,0,:,0][:10, 8])
# Now, do masking manually

# No masking
backbone_feats = model(example_batch)
example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=False)
print(example_out[Output.logrates].size())
print(example_out[Output.logrates][0, :, 0, :][:10, 8]) # extremely spiky prediction

# # With masking
# example_batch[DataKey.spikes][0, :, 0, :, 0][:10] = 0
# backbone_feats = model(example_batch)
# example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=False)
# print(example_out[Output.logrates].size())
# print(example_out[Output.logrates][0, :, 0, :][:10, 8]) # unspiked prediction.
# OK - well if true mask occurs, model appropriately doesn't predict high spike.

# Key symptom - whether or not a spike occurs at a timestep is affecting its own prediction
# example_batch[DataKey.spikes][0, :, 0, :, 0][1] = 0
# backbone_feats = model(example_batch)
# example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=False)
# print(example_out[Output.logrates].size())
# print(example_out[Output.logrates][0, :, 0, :][:10, 8]) # unspiked prediction.


# Masking through model update_batch also seems to work
model.task_pipelines[ModelTask.infill.value].update_batch(example_batch)
print(example_batch['is_masked'][0].nonzero())
backbone_feats = model(example_batch)
example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=True)
# example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=False)
print(example_out[Metric.bps])
print(example_out[Output.logrates].size())
print(example_out[Output.logrates][0, :, 0, :][:10, 8]) # unspiked prediction.


# Ok - so the model is correctly predicting unspiked for masked timesteps.
# Then why is test time evaluation so spiky? Even when we mask?
# Let's check again...
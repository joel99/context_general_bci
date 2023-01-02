#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
from model import transfer_model, logger

from utils import stack_batch, get_wandb_run, load_wandb_run, wandb_query_latest
from utils import prep_plt

# wandb_run = get_wandb_run("maze_med-1j0loymb")
query = "maze_small"
query = "maze_med"
# query = "maze_large"
query = "maze_nlb"
# query = "maze_med_ft"
# query = "maze_small_ft"
# query = "maze_large_ft"
# query = "maze_all"
# query = "rtt_all"
# query = "rtt_all_256"
# query = "rtt_nlb_infill_only"
# query = 'rtt_nlb_07'

# query = "rtt_indy_nlb"
# query = "rtt_indy1"
# query = "rtt_indy2"
# query = "rtt_indy2_noembed"
# query = "rtt_all_sans_add"
# query = "rtt_indy_sans_256_d01"
# query = "rtt_all_256"
# query = "rtt_all_512"
# query = "rtt_indy_loco"

# query = "rtt_loco1"
# query = "rtt_loco1_d3"
# query = "rtt_loco"
# query = "rtt_loco"
query = 'rtt_indy_linear'
# query = 'test'

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, exact=True, allow_running=True)[0]
print(wandb_run.id)

# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='co_bps')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='bps')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
# cfg.dataset.datasets = cfg.dataset.datasets[:1]
cfg.model.task.tasks = [ModelTask.infill]
cfg.model.task.metrics = [Metric.bps, Metric.all_loss]
cfg.model.task.outputs = [Output.logrates, Output.spikes]
cfg.dataset.datasets = cfg.dataset.datasets[-1:]
# cfg.dataset.datasets = ['mc_maze$']
# cfg.dataset.datasets = ['mc_maze_large']
# cfg.dataset.datasets = ['mc_maze_medium']
# cfg.dataset.datasets = ['mc_maze_small']
# cfg.dataset.datasets = ['churchland_maze_jenkins-1']
cfg.dataset.datasets = ['odoherty_rtt-Indy-20161005_06']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170215_02']

print(cfg.dataset.datasets)
dataset = SpikingDataset(cfg.dataset)
dataset.restrict_to_train_set()
dataset.build_context_index()
data_attrs = dataset.get_data_attrs()
model = transfer_model(src_model, cfg.model, data_attrs)
#%%
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
trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='tmp')

heldin_metrics = stack_batch(trainer.test(model, dataloader))

#%%
heldin_outputs = stack_batch(trainer.predict(model, dataloader))
# print(heldin_outputs[Output.rates].max(), heldin_outputs[Output.rates].mean())
# test = heldin_outputs[Output.heldout_rates]
rates = heldin_outputs[Output.rates] # b t c
spikes = heldin_outputs[Output.spikes][:,:,0] # b t a c -> b t c
ax = prep_plt()

num = 3
channel = 2

colors = sns.color_palette("husl", num)

for trial in range(num):
    ax.plot(rates[trial][:,channel], color=colors[trial])

y_lim = ax.get_ylim()[1]
# plot spike raster
for trial in range(num):
    spike_times = spikes[trial,:,channel].nonzero()
    y_height = y_lim * (trial+1) / num
    ax.scatter(spike_times, torch.ones_like(spike_times)*y_height, color=colors[trial], s=10, marker='|')

trial = 20
from scipy.ndimage import gaussian_filter1d
# for channel in range(num):
# #     if channel != 2:
# #         continue
# #     # ax.scatter(np.arange(test.shape[1]), test[0,:,channel], color=colors[channel], s=1)
#     # ax.plot(test[trial][:,channel], color=colors[channel])
#     ax.plot(gaussian_filter1d(test[trial,:,channel], sigma=3), color=colors[channel])

    # smooth the signal with a gaussian kernel

# from scipy import signal
# peaks, _ = signal.find_peaks(test[trial,:,2], distance=4)
# print(peaks)
# print(len(peaks))
# for p in peaks:
#     ax.axvline(p, color='k', linestyle='--')



# relabel xtick unit from 5ms to ms
ax.set_xticklabels(ax.get_xticks() * 5)
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
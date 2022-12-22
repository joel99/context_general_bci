#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

# wandb_run = get_wandb_run("maze_med-1j0loymb")
query = "maze_small"
query = "maze_med"
query = "maze_large"
# query = "maze_med_ft"
# query = "maze_small_ft"
# query = "maze_large_ft"
# query = "maze_all"
query = "rtt_all"
# query = 'rtt_nlb_07'

wandb_run = wandb_query_latest(query, exact=True, allow_running=True)[0]
print(wandb_run.id)

# model, cfg, data_attrs = load_wandb_run(wandb_run, tag='co_bps')
model, cfg, data_attrs = load_wandb_run(wandb_run, tag='val_loss')
print(cfg.dataset.datasets)
cfg.dataset.datasets = cfg.dataset.datasets[:1]
# cfg.dataset.datasets = cfg.dataset.datasets[-1:]
# cfg.dataset.datasets = ['mc_maze$']
# cfg.dataset.datasets = ['mc_maze_medium']
# cfg.dataset.datasets = ['mc_maze_small']
# cfg.dataset.datasets = ['churchland_maze_jenkins-1']
dataset = SpikingDataset(cfg.dataset)
dataset.restrict_to_train_set()
dataset.build_context_index()
data_attrs = dataset.get_data_attrs()
model = transfer_model(model, model.cfg, data_attrs)
#%%
# model.cfg.task.outputs = [Output.heldout_logrates]
model.cfg.task.outputs = [Output.logrates]
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

# heldin_outputs = stack_batch(trainer.test(model, dataloader))
# heldin_outputs = stack_batch(trainer.predict(model, dataloader))
heldin_outputs = stack_batch(trainer.predict(model, dataloader))
#%%
print(data_attrs)
#%%
from utils import prep_plt

print(heldin_outputs.keys())
# print(heldin_outputs[Output.rates].max(), heldin_outputs[Output.rates].mean())

# test = heldin_outputs[Output.heldout_rates]
test = heldin_outputs[Output.rates]
print(test.shape)
num_trials = 5
colors = sns.color_palette("husl", num_trials)
ax = prep_plt()
for trial in range(num_trials):
    plt.plot(test[trial][:,8])
    # plt.plot(test[trial][:,10])
    # plt.plot(test[trial][:,11])
    # plt.plot(test[trial][:,12])
    # plt.plot(test[trial][:,30])
    # ax.plot(test[trial][:,40], color=colors[trial])
    # plt.plot(test[trial][:,100])
    # plt.plot(test[trial][:,65])

# plt.plot(test[0,:,0])
print("done")

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
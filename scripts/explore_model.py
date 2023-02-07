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

# wandb_run = get_wandb_run("maze_med-1j0loymb")
query = "maze_small"
query = "maze_med"
# query = "maze_large"
query = "maze_nlb"
# query = "maze_med_ft"
# query = "maze_small_ft"
# query = "maze_large_ft"
# query = "maze_all_256"
# query = "maze_all"

query = "maze_jenkins_stitch"
# query = "maze_nlb_stitch_out"
# query = "maze_nlb"

# query = 'maze_med_20'

# query = "rtt_all"
# query = "rtt_all_256"
# query = "rtt_nlb_infill_only"
# query = 'rtt_nlb_07'
# query = 'rtt_nlb_pt'

# query = "rtt_indy_nlb"
# query = "rtt_indy_nlb_stitch"
# query = "rtt_indy1"
# query = "rtt_indy2"
# query = "rtt_indy2_noembed"
# query = "rtt_all_sans_add"
# query = "rtt_indy_sans_256_d01"
# query = "rtt_indy_stitch"
query = "rtt_all_256"
query = "rtt_all_parity2"
query = "st_rtt_16"

query = 'factor_maze_8'
query = "rtt_causal"
# query = "factor_rtt_16"
# query = "rtt_all_512"
# query = "rtt_token_padded_nostitch"

# query = 'test'

# query = 'rtt_loco_d2'
# query = 'rtt_loco_512'
# query = 'rtt_loco_256_stitch'
# query = 'test'

# query = 'nitschke_token'
# query = 'nitschke'
# query = 'nitschke_single'

# query = 'gallego_token'
# query = 'gallego'
# query = 'gallego_chewie'
# query = 'gallego_chewie_single'

# query = 'pitt_single'
# query = 'pitt'
# query = 'pitt_obs'
# query = 'pitt_20'
# query = 'pitt_obs_20'
# query = 'pitt_obs_causal'

# query = 'ks'
# query = 'ks_ctx'

# query = 'jenkins_misc'
# query = 'reggie_misc'

query = "rtt_5_factor_1_multi"
# query = "rtt_maze_5_factor_2"

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, exact=True, allow_running=True)[0]
print(wandb_run.id)

# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='co_bps')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='bps')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
# cfg.dataset.datasets = cfg.dataset.datasets[:1]
# cfg.model.task.tasks = [ModelTask.infill]
cfg.model.task.metrics = [Metric.bps]
# cfg.model.task.metrics = [Metric.bps, Metric.all_loss]
cfg.model.task.outputs = [Output.logrates, Output.spikes]
print(cfg.dataset.datasets)
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
# if 'rtt' in query:
#     cfg.dataset.datasets = ['odoherty_rtt-Indy-20161005_06']
#     # cfg.dataset.datasets = ['odoherty_rtt-Indy-20161014_04']
# if 'gallego' in query:
#     cfg.dataset.datasets = ['Chewie_CO_20150313']
#     cfg.dataset.datasets = ['Mihili_CO_20140304']
# if 'nitschke' in query:
#     cfg.dataset.datasets = ['churchland_misc_nitschke-1D8KYfy5IwMmEZaKOEv-7U6-4s-7cKINK']
# if 'pitt' in query:
#     cfg.dataset.datasets = ['CRS02bHome.data.00437']

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

# dataloader = get_dataloader(dataset)
dataloader = get_dataloader(dataset, batch_size=4)
#%%
print(query)
heldin_metrics = stack_batch(trainer.test(model, dataloader))
heldin_outputs = stack_batch(trainer.predict(model, dataloader))

#%%
# print(heldin_outputs[Output.rates].max(), heldin_outputs[Output.rates].mean())
# test = heldin_outputs[Output.heldout_rates]
rates = heldin_outputs[Output.rates] # b t c

spikes = [rearrange(x, 't a c -> t (a c)') for x in heldin_outputs[Output.spikes]]
ax = prep_plt()

num = 20
num = 5
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
# trial = 15
# trial = 17
# trial = 18
# trial = 80
trial = 85
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
# ax.set_xlim(0, 50)
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
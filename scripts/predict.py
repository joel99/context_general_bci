#%%
from collections import defaultdict
from typing import Dict, List
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nlb_tools.make_tensors import save_to_h5

# Load BrainBertInterface and SpikingDataset to make some predictions
from model import BrainBertInterface
from data import SpikingDataset, DataAttrs
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from contexts import context_registry
from copy import deepcopy

from utils import get_latest_ckpt_from_wandb_id, get_wandb_run, load_wandb_run

dataset_name = 'mc_rtt'
dataset_name = 'mc_maze$'

wandb_id = "maze_nlb_ft-umbki5uw"
wandb_id = "maze_jenkins_only_to_med_ft-ihy0h4yv"
wandb_id = "maze_all_med_ft-2iln5gpm"
wandb_id = "maze_med_ft-qetjfh2b"
wandb_run = get_wandb_run(wandb_id)
heldout_model, cfg, data_attrs = load_wandb_run(wandb_run, tag='val_co_bps')
heldout_model.cfg.task.outputs = [Output.heldout_logrates]

cfg.dataset.data_keys = [DataKey.spikes]
cfg.dataset.datasets = [dataset_name] # do _not_ accidentally set this as just a str

dataset = SpikingDataset(cfg.dataset)
test_dataset = deepcopy(dataset)
dataset.restrict_to_train_set()
dataset.build_context_index()
test_dataset.subset_by_key(['test'], key='split')
test_dataset.build_context_index()

base_run = get_wandb_run(cfg.init_from_id)
heldin_model, *_ = load_wandb_run(base_run)
heldin_model.cfg.task.outputs = [Output.logrates]
#%%
def get_dataloader(dataset: SpikingDataset, batch_size=100, num_workers=1, **kwargs) -> DataLoader:
    # Defaults set for evaluation on 1 GPU.
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collater_factory()
    )

dataloader = get_dataloader(dataset)
test_dataloader = get_dataloader(test_dataset)

trainer = pl.Trainer(gpus=1, default_root_dir='tmp')

def stack_batch(batch_out: List[Dict[str, torch.Tensor]]):
    out = defaultdict(list)
    for batch in batch_out:
        for k, v in batch.items():
            out[k].append(v)
    for k, v in out.items():
        out[k] = torch.cat(v)
    return out

heldin_outputs = stack_batch(trainer.predict(heldin_model, dataloader))
heldout_outputs = stack_batch(trainer.predict(heldout_model, dataloader))
test_heldin_outputs = stack_batch(trainer.predict(heldin_model, test_dataloader))
test_heldout_outputs = stack_batch(trainer.predict(heldout_model, test_dataloader))
# print(heldin_outputs[Output.rates].max(), heldin_outputs[Output.rates].mean())
# print(heldout_outputs[Output.heldout_rates].max(), heldout_outputs[Output.heldout_rates].mean())
# print(test_heldout_outputs[Output.heldout_rates].max(), test_heldout_outputs[Output.heldout_rates].mean())

#%%
print(heldin_outputs[Output.rates].shape)
test = heldin_outputs[Output.rates].squeeze(2).numpy()
test = test_heldin_outputs[Output.rates].squeeze(2).numpy()
test = test_heldout_outputs[Output.heldout_rates].numpy()
for trial in range(len(test)):
    plt.plot(test[trial,:,0])
    if trial > 5:
        break
# plt.plot(test[0,:,0])
print("done")
#%%
# Create spikes for NLB submission https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb
dataset_name = dataset.cfg.datasets[0]
suffix = '' # no suffix needed for 5ms submissions
output_dict = {
    dataset_name + suffix: {
        'train_rates_heldin': heldin_outputs[Output.rates].squeeze(2).numpy(),
        'train_rates_heldout': heldout_outputs[Output.heldout_rates].numpy(),
        'eval_rates_heldin': test_heldin_outputs[Output.rates].squeeze(2).numpy(),
        'eval_rates_heldout': test_heldout_outputs[Output.heldout_rates].numpy(),
    }
}
print(output_dict.keys())
print(output_dict[dataset_name + suffix].keys()) # sanity check
# for rtt, expected shapes are 1080 / 272, 120, 98 / 32
print(output_dict[dataset_name + suffix]['train_rates_heldin'].shape) # should be trial x time x neuron
# print(output_dict[dataset_name + suffix]['train_rates_heldout'])
save_to_h5(output_dict, "submission.h5")

#%%
from collections import defaultdict
from typing import Dict, List
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from copy import deepcopy

from nlb_tools.make_tensors import save_to_h5

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from contexts import ContextInfo, context_registry
from data import SpikingDataset, DataAttrs
from model import BrainBertInterface

from utils import stack_batch, get_wandb_run, load_wandb_run, wandb_query_latest

wandb_run = wandb_query_latest('maze_med')
wandb_run = get_wandb_run('maze_med-3usppdke')

model, cfg, data_attrs = load_wandb_run(wandb_run)

#%%

dataset = SpikingDataset(cfg.dataset)
dataset.restrict_to_train_set()
dataset.build_context_index()

model.cfg.task.outputs = [Output.logrates]
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
trainer = pl.Trainer(gpus=1, default_root_dir='tmp')

heldin_outputs = stack_batch(trainer.predict(model, dataloader))
print(heldin_outputs[Output.rates].max(), heldin_outputs[Output.rates].mean())

#%%
test = heldin_outputs[Output.rates].squeeze(2).numpy()
for trial in range(test.shape[0]):
    plt.plot(test[trial,:,0])
    if trial > 5:
        break
# plt.plot(test[0,:,0])
print("done")
#%%
# CPU testing harness
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
# from model import transfer_model, logger

from analyze_utils import stack_batch, load_wandb_run
from analyze_utils import prep_plt, get_dataloader
from utils import wandb_query_experiment, get_wandb_run, wandb_query_latest


from model_decode import transfer_model
pl.seed_everything(0)

UNSORT = True
# UNSORT = False

ROBUST_RUN = 'session_cross_noctx-89e73b3s'
DATASET_WHITELIST = [
    "odoherty_rtt-Indy-20160407_02",
    "odoherty_rtt-Indy-20170131_02",
    "odoherty_rtt-Indy-20160627_01",
]

# trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
trainer = pl.Trainer(accelerator='cpu', default_root_dir='./data/tmp')

run = get_wandb_run(ROBUST_RUN)
src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
dataset_name = DATASET_WHITELIST[0] # drop wandb ID
cfg.dataset.datasets = [dataset_name]
cfg.dataset.exclude_datasets = []

dataset = SpikingDataset(cfg.dataset)
dataset.subset_split(splits=['eval'])
dataset.build_context_index()
data_attrs = dataset.get_data_attrs()
cfg.model.task.tasks = [ModelTask.kinematic_decoding]
cfg.model.task.outputs = [Output.behavior_pred]
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
dataloader = get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

# script = torch.jit.script(model, next(iter(dataloader)))
# script.save(PATH)
# model = torch.jit.load(PATH)

# TODO test with a bigger model - an actual tuned CRS model with
# 2x array and longer history as well..

# Setup timing harness
import time

loop_times = []
mode = 'cpu'
mode = 'gpu'
if mode == 'gpu':
    model = model.to('cuda:0')
with torch.no_grad():
    for batch in dataloader:
        start = time.time()
        import pdb;pdb.set_trace()
        if mode == 'gpu':
            for k in batch:
                batch[k] = batch[k].to('cuda:0')
        out = model(batch)
        out = model.task_pipelines[ModelTask.kinematic_decoding.value](
            batch,
            out,
            compute_metrics=False,
            eval_mode=True
        )[Output.behavior_pred]
        if mode == 'gpu':
            out = out.to('cpu')
        end = time.time()
        loop_times.append(end - start)
        print(f'Loop time: {end - start:.4f}')

print(f"Avg: {np.mean(loop_times)*1000:.4f}ms, Std: {np.std(loop_times) * 1000:.4f}ms")
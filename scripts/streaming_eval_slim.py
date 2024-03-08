# %%
# Notebook for evaluating performance at different streaming lengths (quant)
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from einops import rearrange
import pytorch_lightning as pl
from sklearn.metrics import r2_score

# Run this block to eval on minival
FALCON_MINIVAL = False
FALCON_MINIVAL = True

if FALCON_MINIVAL:
    from context_general_bci.utils import suppress_default_registry
    suppress_default_registry()
    from context_general_bci.contexts.context_registry import context_registry
    from context_general_bci.contexts.context_info import FalconContextInfo, ExperimentalTask
    context_registry.register([
        *FalconContextInfo.build_from_dir('./data/h1/minival', task=ExperimentalTask.falcon, suffix='minival'),
    ])

from context_general_bci.model import BrainBertInterface, transfer_model
from context_general_bci.model_slim import transfer_model as transfer_model_slim

from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import (
    Metric,
    Output,
    DataKey,
    MetaKey,
)

from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id, to_device
from context_general_bci.analyze_utils import (
    stack_batch,
    load_wandb_run,
    prep_plt,
    get_dataloader,
)
from context_general_bci.streaming_utils import (
    precrop_batch,
    postcrop_batch,
)

query = 'h1_v2-sweep-h1_fine_grained_discrete-4j1mi057'
query = 'h1_v2-sweep-h1_fine_grained_discrete-v6luzk35'
query = 'h1_nopool_cross-sweep-h1_fine_grained_discrete-rasu7u1w'
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# tag = 'val_loss'
tag = "val_kinematic_r2"
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
# Parse epoch, format is `val-epoch=<>-other_metrics.ckpt`
ckpt_epoch = int(str(ckpt).split("-")[1].split("=")[1])

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
]
target = [
    'falcon_FALCONH1.*',
]

cfg.dataset.datasets = target
cfg.dataset.exclude_datasets = []
cfg.dataset.eval_datasets = []

stream_buffer_s = cfg.dataset.max_length_ms / 1000
cfg.dataset.max_length_ms = 20000 # Some large number to serve full data
dataset = SpikingDataset(cfg.dataset)

prompt = None
pl.seed_everything(0)
data_attrs = dataset.get_data_attrs()
if not FALCON_MINIVAL:
    train, val = dataset.create_tv_datasets()
    dataset = val
print("Eval length: ", len(dataset))
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model_slim = transfer_model_slim(src_model, cfg.model, data_attrs)
model.eval()
model_slim.eval()
model = model.to("cuda")
model_slim = model_slim.to("cuda")

# %%
labels = ['x', 'y', 'z', 'rx', 'g1', 'g2', 'g3']
def eval_model(
    model: BrainBertInterface,
    dataset: SpikingDataset,
):
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0)

    outputs = []
    slim_outputs = []
    for batch in dataloader:
        print('Batch')
        batch = to_device(batch, "cuda")

        if stream_buffer_s:
            timesteps = batch[DataKey.time].max() + 1 # number of distinct timesteps
            buffer_steps = int(stream_buffer_s * 1000 // cfg.dataset.bin_size_ms)
            stream_output = []
            stream_slim_output = []
            for end_time_exclusive in range(1, timesteps + 1): # +1 because range is exlusive
                stream_batch = deepcopy(batch)
                stream_batch = precrop_batch(stream_batch, end_time_exclusive) # Keep to end_time
                crop_suffix = max(end_time_exclusive - buffer_steps, 0)
                stream_batch = postcrop_batch(stream_batch, crop_suffix) # Take last STREAM_BUFFER_S
                parity_batch = {k: v for k, v in stream_batch.items() if k in [
                    DataKey.spikes,
                    DataKey.time,
                    DataKey.position,
                    DataKey.bhvr_vel,
                    DataKey.bhvr_mask,
                ] or not isinstance(k, DataKey)}
                output = model.predict_simple_batch( # Match streaming API _exactly_, see `rtndt.accelerators` call in CLIMBER
                    parity_batch,
                    last_step_only=True,
                )
                # Split time dimension
                stride = len(parity_batch[DataKey.position].unique())
                reconstructed_flat_spikes = rearrange(
                    parity_batch[DataKey.spikes],
                    'b (time space) patch 1 -> b time (space patch) 1',
                    space=stride
                )
                session_id = parity_batch[MetaKey.session]
                output_slim = model_slim(reconstructed_flat_spikes, session_id, last_step_only=True)
                stream_slim_output.append(output_slim)
                stream_output.append(output)
            stream_total = stack_batch(stream_output) # concat behavior preds
            stream_slim_total = stack_batch(stream_slim_output)
            outputs.append(stream_total)
            slim_outputs.append(stream_slim_total)
        else:
            output = model.predict_simple_batch(
                batch,
                last_step_only=False,
            )
            outputs.append(output)
    outputs = stack_batch(outputs)
    slim_outputs = stack_batch(slim_outputs)

    print(f"Checkpoint: {ckpt_epoch} (tag: {tag})")
    prediction = outputs[Output.behavior_pred].cpu()
    target = outputs[Output.behavior].cpu()
    if stream_buffer_s:
        if Output.behavior_mask in outputs:
            valid = outputs[Output.behavior_mask][:, 0].cpu().bool()
        else:
            valid = torch.ones(prediction.shape[0], dtype=torch.bool)
        is_student = valid
        loss = 0.
    print(f"Computing R2 on {valid.sum()} of {valid.shape} points")
    mse = torch.mean((target[valid] - prediction[valid]) ** 2, dim=0).mean()
    r2_student = r2_score(target[valid], prediction[valid], multioutput="variance_weighted")
    print(f"MSE: {mse:.3f}")
    print(f"R2 (Weighted): {r2_student:.3f}")

    # Get reported metrics
    history = wandb_run.scan_history(
        keys=[
            'val_kinematic_r2',
            'val_loss',
            'epoch',
        ]
    )
    history = pd.DataFrame(history)
    history = history.dropna(subset=["epoch"])
    history.loc[:, "epoch"] = history["epoch"].astype(int)
    ckpt_rows = history[history["epoch"] == ckpt_epoch]
    # Cast epoch to int or 0 if nan, use df loc to set in place
    # Get last one
    reported_r2 = ckpt_rows[f"val_{Metric.kinematic_r2.name}"].values[-1]
    reported_loss = ckpt_rows[f"val_loss"].values[-1]
    print(f"Reported R2: {reported_r2:.3f}")
    print(f"Reported Loss: {reported_loss:.3f}")
    return outputs, target, prediction, is_student, valid, r2_student, mse


(outputs, target, prediction, is_student, valid, r2_student, mse) = eval_model(
    model, dataset
)

# %%
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)
palette = sns.color_palette(n_colors=2)
colors = [palette[0] if is_student[i] else palette[1] for i in range(len(is_student))]
alpha = [0.1 if is_student[i] else 0.8 for i in range(len(is_student))]
ax.scatter(target[valid], prediction[valid], s=3, alpha=alpha)
ax.set_xlabel("True")
ax.set_ylabel("Pred")
ax.set_title(f"{query} R2: {r2_student:.2f}")


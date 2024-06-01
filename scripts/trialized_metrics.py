#%%
# Eval the trialized model for various histories
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

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.analyze_utils import stack_batch, get_dataloader, load_wandb_run, prep_plt, get_best_ckpt_from_wandb_id, get_run_config
from context_general_bci.utils import wandb_query_experiment, get_wandb_run


from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from context_general_bci.falcon_decoder import NDT2Decoder
# from scripts.falcon_local import run_evaluate

pl.seed_everything(0)

# argparse the eval set
import sys
import argparse

num_workers = 4 # for main eval block.
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    VARIANT = 'm2'
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
    parser.add_argument(
        "--trialized", "-t", action='store_true'
    )
    args = parser.parse_args()
    VARIANT = args.variant
    TRIALIZED_EVAL = args.trialized # still requires hardcode override in package - manually do this
eval_set = VARIANT

NDT2_EXPERIMENT_MAP = {
    "h1": "falcon/h1/h1_100",
    "m1": "falcon/m1/m1_100",
    "m2": "falcon/m2/m2_100",
}

eval_paths = Path('./data/falcon_metrics')
eval_paths.mkdir(exist_ok=True, parents=True)

queries = [
    ""
]

runs = []

ndt2_run_df = pd.DataFrame({
    'history': [200, 500, 800, 1000, 2000, 4000]
})

ckpts = {
    'h1': '/home/joy47/projects/stability-benchmark/local_data/ndt2_h1_sample.pth',
    'm1': '/home/joy47/projects/stability-benchmark/local_data/ndt2_m1_sample.pth',
    'm2': '/home/joy47/projects/stability-benchmark/local_data/ndt2_m2_sample.pth'
}
norms = {
    'h1': '/home/joy47/projects/stability-benchmark/local_data/ndt2_zscore_h1.pt',
    'm1': '/home/joy47/projects/stability-benchmark/local_data/ndt2_zscore_m1.pt',
    'm2': '/home/joy47/projects/stability-benchmark/local_data/ndt2_zscore_m2.pt'
}
    

eval_metrics_path = eval_paths / f"{eval_set}_{'trialized' if TRIALIZED_EVAL else 'continual'}_ndt2_trialized.csv"
eval_metrics = {}
ndt2_run_df['eval_r2'] = 0.
for idx, run_row in ndt2_run_df.iterrows():
    run = get_wandb_run(run_row.id, wandb_project="context_general_bci")
    ckpt = ckpts[VARIANT]
    zscore_pth = norms[VARIANT]
    split = run_row.eval_set
    evaluator = FalconEvaluator(
        eval_remote=False,
        split=split,
        verbose=True
    )

    task = getattr(FalconTask, split)
    config = FalconConfig(task=task)
    max_bins = run_row['history']
    decoder = NDT2Decoder(
        task_config=config,
        model_ckpt_path=ckpt,
        model_cfg_stem=NDT2_EXPERIMENT_MAP[VARIANT],
        zscore_path=zscore_pth,
        max_bins=max_bins,
        dataset_handles=[x.stem for x in evaluator.get_eval_files(phase='test')],
        batch_size=1 if TRIALIZED_EVAL else 8,
    )
    payload = evaluator.evaluate(decoder, phase='test')
    result = payload['result'][0][f'test_split_{split}']
    eval_r2 = result['Held Out R2 Mean']
    if 'heldin_eval_r2' not in ndt2_run_df.columns:
        ndt2_run_df['heldin_eval_r2'] = 0.
    heldin_eval_r2 = result['Held In R2 Mean']
    ndt2_run_df.at[idx, 'eval_r2'] = eval_r2
    ndt2_run_df.at[idx, 'heldin_eval_r2'] = heldin_eval_r2
    # Copy down all other metrics
    del result['Held Out R2 Mean']
    del result['Held In R2 Mean']
    for key, val in result.items():
        ndt2_run_df.at[idx, key] = val
    print(ndt2_run_df.iloc[idx])

#%%
ndt2_run_df.to_csv(eval_metrics_path, index=False)

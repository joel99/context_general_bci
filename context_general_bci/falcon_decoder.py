r"""
    NDT2 wrapper. Not for running.
"""
from typing import List, Union, Dict
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from einops import rearrange

from hydra import compose, initialize_config_module

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder

from context_general_bci.utils import suppress_default_registry
suppress_default_registry()
from context_general_bci.config import RootConfig, propagate_config, DataKey, MetaKey
from context_general_bci.dataset import DataAttrs, ContextAttrs
from context_general_bci.subjects import SubjectName
from context_general_bci.contexts.context_info import ExperimentalTask
from context_general_bci.model import load_from_checkpoint
from context_general_bci.model_slim import transfer_model

def format_array_name(subject: str):
    return f'FALCON{subject}-M1'

class NDT2Decoder(BCIDecoder):
    r"""
        For the FALCON challenge
    """

    def __init__(
            self,
            task_config: FalconConfig,
            model_ckpt_path: Union[str, List[str]],
            model_cfg_stem: str,
            zscore_path: str,
            max_bins: int,
            dataset_handles: List[str] = [],
            batch_size: int = 1,
            force_static_key: str = '' # If true, ignore data attrs and just load the only key fit in the single session model (for oracle single day baselines)
        ):
        r"""
            model_ckpt_path: either individual path, or dict of paths for each session key (single session models)
            Loading NDT2 requires both weights and model config. Weight loading through a checkpoint is standard.
            Model config is typically stored on wandb, but this is not portable enough. Instead, directly reference the model config file.
        """
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.exp_task = getattr(ExperimentalTask, f'falcon_{task_config.task.name}')
        try:
            initialize_config_module(
                config_module="context_general_bci.config",
                job_name="falcon",
                version_base="1.3",
            )
        except:
            print('Hydra Initialize failed, assuming this is not the first decoder.')
        exp_stem, proper_stem = model_cfg_stem.split('/')[:-1], model_cfg_stem.split('/')[-1]
        exp_stem = '/'.join(exp_stem)
        override_path = f"+exp/{exp_stem}={proper_stem}"
        cfg: RootConfig = compose(config_name="config", overrides=[override_path])

        propagate_config(cfg)
        pl.seed_everything(seed=cfg.seed)

        cfg.model.task.decode_normalizer = zscore_path
        self.subject = getattr(SubjectName, f'falcon_{task_config.task.name}')
        
        if force_static_key:
            sessions = [force_static_key]
            self.forced = True
        else:
            sessions = sorted([self._task_config.hash_dataset(handle) for handle in dataset_handles])
            sessions = [self.reduce_tag(i) for i in sessions]
            self.forced = False
        def make_context(sessions):
            context_idx = {
                MetaKey.array.name: [format_array_name(self.subject)],
                MetaKey.subject.name: [self.subject],
                MetaKey.session.name: sessions,
                MetaKey.task.name: [self.exp_task],
            }
            return DataAttrs.from_config(cfg.dataset, context=ContextAttrs(**context_idx))
        if isinstance(model_ckpt_path, list):
            assert not self.forced, "Forcing static key not supported with multiple models."
            assert self.batch_size == 1, "Batch size 1 required for multiple models."
            # need to load a dictionary of single-session models - but how do I know which session needs to be loaded?
            # extract the training info
            self.models = {}
            for i in range(len(model_ckpt_path)):
                print(f'Loading: {model_ckpt_path[i]}')
                test = torch.load(model_ckpt_path[i])
                data_attrs = test['hyper_parameters']['data_attrs']
                sessions = test['hyper_parameters']['data_attrs'].context.session
                hidden_size = test['hyper_parameters']['cfg']['hidden_size']
                arch = test['hyper_parameters']['cfg']['arch']
                npt = test['hyper_parameters']['cfg']['neurons_per_token']
                local_cfg = deepcopy(cfg)
                local_cfg.model.hidden_size = hidden_size
                local_cfg.model.arch = arch
                local_cfg.model.neurons_per_token = npt
                local_cfg.model.task.tasks = test['hyper_parameters']['cfg']['task']['tasks']
                local_cfg.model.decoder_layers = test['hyper_parameters']['cfg']['decoder_layers']
                local_cfg.model.task.decode_separate = test['hyper_parameters']['cfg']['task']['decode_separate']
                local_cfg.model.task.decode_strategy = test['hyper_parameters']['cfg']['task']['decode_strategy']
                # breakpoint()
                propagate_config(local_cfg)
                model = load_from_checkpoint(model_ckpt_path[i], cfg=local_cfg.model, data_attrs=data_attrs)
                model = transfer_model(model, local_cfg.model, data_attrs, batch_size=batch_size)
                for s in sessions:
                    self.models[self.reduce_tag(s)] = model # we'll load to GPU when we need to predict, don't forget to eval()
            self.model = None
        else:
            data_attrs = make_context(sessions)
            model = load_from_checkpoint(model_ckpt_path, cfg=cfg.model, data_attrs=data_attrs)
            model = transfer_model(model, cfg.model, data_attrs, batch_size=batch_size)
            self.model = model.to('cuda:0')
            self.model.eval()
            self.models = {}

        assert task_config.bin_size_ms == cfg.dataset.bin_size_ms, "Bin size mismatch, transform not implemented."
        
        self.observation_buffer = torch.zeros((
            max_bins, 
            self.batch_size,
            task_config.n_channels
        ), dtype=torch.uint8, device='cuda:0')
    
    def set_batch_size(self, batch_size: int):
        super().set_batch_size(batch_size)
        self.observation_buffer = torch.zeros((
            self.observation_buffer.shape[0],
            batch_size,
            self.observation_buffer.shape[2]
        ), dtype=torch.uint8, device='cuda:0')
        
    def reduce_tag(self, dataset_tag: str):
        if self._task_config.task == FalconTask.m2:
            def remap_run(run): # Run1_20201019 -> 2020-10-19-Run1
                # Remove explicit run
                if run.startswith('Run'):
                    run_date = run.split('_')[1]
                    return f'{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}' # -{run_str}'
                else:
                    run_date = '-'.join(run.split('-')[:-1])
                    return run_date
            dataset_tag = remap_run(dataset_tag)
        elif self._task_config.task == FalconTask.h1:
            dataset_tag = dataset_tag.split('_set_')[0]
        return dataset_tag

    def reset(self, dataset_tags: List[Path] = [""]):
        self.set_steps = 0
        self.observation_buffer.zero_()
        dataset_tags = [self._task_config.hash_dataset(dset.stem) for dset in dataset_tags]
        meta_keys = []
        if self.forced:
            self.meta_key = torch.zeros(len(dataset_tags), device='cuda:0', dtype=torch.long)
        else:
            for dataset_tag in dataset_tags:
                dataset_tag = self.reduce_tag(dataset_tag)
                if (self.models and dataset_tag not in self.models) or (not self.models and dataset_tag not in self.model.data_attrs.context.session):
                    raise ValueError(f"Dataset tag {dataset_tag} not found in calibration sets {self.model.data_attrs.context.session} - did you calibrate on this dataset?")
                if self.models:
                    self.model = self.models[dataset_tag]
                    self.model = self.model.to('cuda:0')
                    self.model.eval()
                    meta_keys.append(0)
                else:
                    meta_keys.append(self.model.data_attrs.context.session.index(dataset_tag))
            self.meta_key = torch.tensor(meta_keys, device='cuda:0')

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            
            return:
                out: (batch, n_dims)
        """
        self.observe(neural_observations)
        decoder_in = rearrange(self.observation_buffer[-self.set_steps:], 't b c -> b t c 1')
        out = self.model(decoder_in[:len(self.meta_key)], self.meta_key)
        return out.cpu().numpy()

    def observe(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            - for timestamps where we don't want predictions but neural data may be informative (start of trial)
        """
        if neural_observations.shape[0] < self.batch_size:
            neural_observations = np.pad(neural_observations, ((0, self.batch_size - neural_observations.shape[0]), (0, 0)))
        self.set_steps += 1
        self.observation_buffer = torch.roll(self.observation_buffer, -1, dims=0)
        self.observation_buffer[-1] = torch.as_tensor(neural_observations, dtype=torch.uint8, device='cuda:0')

    def on_done(self, dones: np.ndarray):
        r"""
            Note we don't reset set_steps because we don't want to kill history of not yet terminated trials in batch.
            We assume we are only being evaluated in file-parallel, so after one trial finishes, predictions from that trial
            is not used until reset.
            
            Note: NDT2 overfits to trial structure. Hence we assume for optimal performance, batch size 1 and always reset when trial switches, to 
            maximize presence of trial structure at eval.
        """
        if dones.any():
            self.set_steps = 0 
        if dones.shape[0] < self.batch_size:
            dones = np.pad(dones, (0, self.batch_size - dones.shape[0]))
        self.observation_buffer[:, dones].zero_()


if __name__ == "__main__":
    print(f"No train/calibration capabilities in {__file__}, use `context_general_bci` codebase.")
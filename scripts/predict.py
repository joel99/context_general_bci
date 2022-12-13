#%%
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nlb_tools.make_tensors import save_to_h5

# Load BrainBertInterface and SpikingDataset to make some predictions
from model import BrainBertInterface
from data import SpikingDataset
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from contexts import context_registry
from copy import deepcopy

from utils import get_latest_ckpt_from_wandb_id, get_wandb_run


dataset_name = 'mc_rtt'
context = context_registry.query(alias=dataset_name)

default_cfg = RootConfig()
default_cfg.dataset.bin_size_ms = 5
default_cfg.dataset.max_arrays = min(max(1, len(context.array)), 2)
default_cfg.dataset.data_keys = [DataKey.spikes, DataKey.heldout_spikes, DataKey.test_spikes]

dataset = SpikingDataset(default_cfg.dataset)
dataset.meta_df = dataset.load_session(context.alias)[0]
test_dataset = deepcopy(dataset)
#%%
dataset.restrict_to_train_set()
dataset.build_context_index()
test_dataset.subset_by_key(['test'], key='split')
test_dataset.build_context_index()

# Load the model
wandb_id = "rtt_ft-3163c5n8"
run = get_wandb_run(default_cfg.wandb_project, wandb_id)
co_bps_ckpt = get_latest_ckpt_from_wandb_id(default_cfg.wandb_project, wandb_id, tag="val_co_bps")
heldout_model = BrainBertInterface.load_from_checkpoint(co_bps_ckpt)

base_id = run.config['init_from_id']
base_ckpt = get_latest_ckpt_from_wandb_id(default_cfg.wandb_project, base_id)
heldin_model = BrainBertInterface.load_from_checkpoint(base_ckpt)
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

heldin_outputs = trainer.predict(heldin_model, dataloader)
heldin_rates = heldin_model.unpad_and_transform_rates(heldin_outputs[Output.logrates])
heldout_outputs = trainer.predict(heldout_model, dataloader)
heldout_rates = heldout_model.unpad_and_transform_rates(heldout_outputs[Output.heldout_logrates])
test_heldin_outputs = trainer.predict(heldin_model, test_dataloader)
test_heldin_rates = heldin_model.unpad_and_transform_rates(test_heldin_outputs[Output.logrates])
test_heldout_outputs = trainer.predict(heldout_model, test_dataloader)
test_heldout_rates = heldout_model.unpad_and_transform_rates(test_heldout_outputs[Output.heldout_logrates])

# TODO do they want logrates? no, they want spikes per bin
# Create spikes for NLB submission https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb
dataset_name = dataset.cfg.datasets[0]
suffix = '' # no suffix needed for 5ms submissions
output_dict = {
    dataset_name + suffix: {
        'train_rates_heldin': heldin_rates,
        'train_rates_heldout': heldout_rates,
        'eval_rates_heldin': test_heldin_rates,
        'eval_rates_heldout': test_heldout_rates,
    }
}

print(output_dict.keys())
print(output_dict[dataset_name + suffix].keys()) # sanity check
# for rtt, expected shapes are 1080 / 272, 120, 98 / 32
print(output_dict[dataset_name + suffix]['train_rates_heldin'].shape) # should be trial x time x neuron
print(output_dict[dataset_name + suffix]['train_rates_heldin'])

save_to_h5(output_dict, "submission.h5")

#%%
# TODO implement model.predict to produce outputs you desire
# TODO connect to NLB pipeline and make private test submission
#%%

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from matplotlib import pyplot as plt
import seaborn as sns

# Load BrainBertInterface and SpikingDataset to make some predictions
from model import BrainBertInterface
from data import SpikingDataset
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from contexts import context_registry

from utils import get_latest_ckpt_from_wandb_id, get_wandb_run


dataset_name = 'mc_rtt'
context = context_registry.query(alias=dataset_name)

default_cfg = RootConfig()
default_cfg.dataset.bin_size_ms = 5
default_cfg.dataset.max_arrays = min(max(1, len(context.array)), 2)
default_cfg.dataset.data_keys = [DataKey.spikes, DataKey.heldout_spikes, DataKey.test_spikes]

dataset = SpikingDataset(default_cfg.dataset)
dataset.meta_df = dataset.load_session(context.alias)[0]
dataset.build_context_index()

# Load the model
wandb_id = "rtt_nlb_ft-1jgisl2s"
run = get_wandb_run(default_cfg.wandb_project, wandb_id)
co_bps_ckpt = get_latest_ckpt_from_wandb_id(default_cfg.wandb_project, wandb_id, tag="val_co_bps")
heldout_model = BrainBertInterface.load_from_checkpoint(co_bps_ckpt)

base_id = run.config['init_from_id']
base_ckpt = get_latest_ckpt_from_wandb_id(default_cfg.wandb_project, base_id)
heldin_model = BrainBertInterface.load_from_checkpoint(base_ckpt)
#%%
test_dataset = dataset.get_test_dataset()

train_heldin_predictions = heldin_model.predict(dataset)[Output.rates]
train_heldout_predictions = heldout_model.predict(dataset)[Output.heldout_rates]
test_heldin_predictions = heldin_model.predict(dataset, heldout=True)[Output.rates]

# TODO get test set spikes into NLB
# TODO implement model.predict to produce outputs you desire
# TODO convert current "predict" into a generic predict API per einops and also using dataloader
# TODO connect to NLB pipeline and make private test submission
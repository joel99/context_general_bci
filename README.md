# context_general_bci
Codebase for Neural Data Transformer 2. This should provide everything you need to run experiments reported in the [NDT2 manuscript](https://www.biorxiv.org/content/10.1101/2023.09.18.558113v1) (i.e. the RTT results) and the [FALCON project](https://snel-repo.github.io/falcon). This codebase is not reduced to the minimal skeleton needed to reproduce; other experimental code is included. Most of the following instructions refer to steps for reproducing the main manuscript.


## Getting started

### Code Requirements
We recommend setting up with:
```
conda create --name onnx python=3.9
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu117
```

This core setup does not include several dataset-specific dependencies (but training with said data will fail without them). These extra dependencies can be installed with:
```
pip install -r additional_requirements.txt
```

### Data Setup
Datasets and checkpoints are expected to go under `./data`, please create or symlink that.

Install the public datasets with the following command; for troubleshooting, comments in individual modules under `context_general_bci/tasks` have specific instructions.
```
. install_datasets.sh
```

### Running an experiment
Logging is done on wandb, which should be set up before runs are launched (please follow wandb setup guidelines and configure your user in `config_base`.)
Provided all paths are setup, start a given run with:
`python run.py +exp/<EXP_SET>=<exp>`.
e.g. to run the experiment configured in `context_general_bci/config/exp/arch/base/f32.yaml`: `python run.py +exp/arch/base=f32`.

You can launch on slurm via `sbatch ./launch.sh +exp/<EXPSET>=<exp>`, or any of the `launch` scripts. The directives should be updated accordingly. Please note there are several config level mechanisms (`inherit_exp`, `inherit_tag`) in place to support experiment loading inheritance, that is tightly coupled to the wandb checkpoint logging system.
A whole folder can be launched through slurm with `python launch_exp.py -e ./context_general_bci/config/exp/arch/base`.
Note for slurm jobs, I trigger the necessary env loads with a `load_env.sh` script located _outside_ this repo, which then point back into the samples provided (`load_env, load_env_crc.sh`), feel free to edit these to match your local environment needs.

Configurations for hyperparameter sweeping can be preset, see e.g. `exp/arch/tune_hp`. Only manual grid or random searches are currently implemented.

## Supporting a new task/dataset.
Implementing a new task or dataset involves a few steps, since the codebase generally requires metadata registration to provide data to the model. This is true even if the model doesn't use the metadata. The process is as follows:
1. In `context_general_bci/subjects`, register a new subject name and class. They can e.g. be added to `pitt_chicago.py` or `nlb_monkeys.py`. Subjects hold information about array geometry or quality; by default you can dictate `SortedArray` for a minimally structured class that requires max channel count only.
2. In `context_general_bci/tasks`, register a new `ExperimentalTask` and `ExperimentalTaskLoader`, the former is an enum and the latter defines the preprocessing needed to parse the datafile into a set of trials for the model to operate on. If the data item size is too large for model context, the default dataloader will randomly crop out a subset of the trial on each iteration based on dataset configuration - be careful of this effect when trying for consistent evaluation.
3. In `context_general_bci/contexts`, register a new `ContextInfo` subclass. A context class contains the information necessary to identify which subjects, tasks are relevant for a given datafile. Each datafile is assumed to correspond to a single experimental session.
4. Recommended: If performing decoding, output dimension normalization can be computed globally in a separate script (e.g. `compute_normalizer.py`) and specified in `ModelConfig.task.decode_normalizer`. Per-item normalization is also supported via `DatasetConfig.z_score`.
5. Specify an experimental configuration: this is a yaml file that specifies various hyperparameters, composed into the global configuration in `config_base.py` suing the [Hydra](https://hydra.cc/) library. See the many configurations in `context_general_bci/exp/` for examples. Looking through `config_base` to consider which particular HPs are relevant to your task is recommended.
6. Sweeping: NDT2 codebase supports basic SLURM-based hyperparameter sweeping. This is done by specifying a `sweep_cfg` key in the experimental configuration; which will pull sweep settings from `hp_sweep_space.py`. See `context_general_bci/exp/falcon/h1/` for examples.
7. Evaluation: Different inference procedures are scattered throughout `scripts`. Basic utilities are available for pulling a checkpoint and config related to a specific W&B ID, but for proper deployment additional code will be needed to handle the specifics of your application.

## Decoding applications (FALCON)
While NDT2 is initially described as a self-supervised pretraining method, you may be interested in its decoding application. Please see the [FALCON-challenge codebase](https://github.com/snel-repo/falcon-challenge) for an example flow for preparaing NDT2 as a decoder. Configs used to produce FALCON baselines are available under `config/exp/falcon`. For example, to reproduce the M2 checkpoints, run `python run.py +exp/falcon/m2_chop_2s`.
The config used for the Joint baselines are:
- H1: [falcon/h1_100](https://wandb.ai/joelye9/context_general_bci/runs/8l13b722?nw=nwuserjoelye9)
- M1: [falcon/m1_100_continual](https://wandb.ai/joelye9/context_general_bci/runs/b6bzlvc1?nw=nwuserjoelye9)
- M2: [falcon/m2_100_continual](https://wandb.ai/joelye9/context_general_bci/runs/rbajwckl/overview?nw=nwuserjoelye9)
In the event the specific config files are changed, the exact git state and configurations when these configs were declared are available in the linked Weights and Bias pages.
Checkpoints for these runs are available [here](https://drive.google.com/drive/u/0/folders/1ijvmPcbyjHlLEoWWSSZrNm1bBojGgrZG).

## Checkpoints
Given the relatively lightweight training process we recommend training your own model to then analyze. This will require overwriting the default wandb settings with your own.
Note analysis scripts aren't build around manual checkpoint path loading; but `model.load_from_checkpoint(<download_path>)` can be used with public checkpoints.

We provide two example checkpoints:
- one from [human data (~100 hours)](https://wandb.ai/joelye9/context_general_bci/runs/j7mq2snc/overview): [Checkpoint](https://drive.google.com/file/d/1sdZAVIlH2CCh856BjQlEdKpRUO0IUGdU/view?usp=drive_link)
- one from [Indy multisession RTT](https://wandb.ai/joelye9/context_general_bci/runs/uych1wae?workspace=user-joelye9): [Checkpoint](https://drive.google.com/file/d/1hhC4n1UyiYjCcv1nlO6ESljNhr8qVlUF/view?usp=share_link).

Checkpoints for the FALCON baselines are in this folder: [FALCON-checkpoints](https://drive.google.com/drive/folders/1ijvmPcbyjHlLEoWWSSZrNm1bBojGgrZG?usp=sharing).


## Other Notes
- The codebase was actually developed in Python 3.10 but this release uses 3.9 for compatibility with `onnx`. Exact numerical reproduction of paper results is not asserted, but please file an issue if large discrepancies with reported results arise.
- Check out ./scripts/figures/` and this [wandb workspace](https://wandb.ai/joelye9/context_general_bci) to see how the results were generated.




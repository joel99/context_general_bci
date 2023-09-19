# context_general_bci
Codebase for Neural Data Transformer 2. This should provide everything you need to run the reported experiments on public datasets, i.e. the main RTT results. This codebase is not reduced to the minimal skeleton needed to operate on public datasets -- please excuse the extraneous files related to Pitt datasets.

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

## Checkpoints
Given the relatively lightweight training process we recommend training your own model to then analyze. This will require overwriting the default wandb settings with your own.
Note analysis scripts aren't build around manual checkpoint loading; but `model.load_from_checkpoint(<download_path>)` can be used with public checkpoints.

Two example checkpoints are provided:
- one from [task scaling](https://wandb.ai/joelye9/context_general_bci/runs/ydv48n02?workspace=user-joelye9): [Checkpoint](https://drive.google.com/file/d/18UgglFKPu6ev5Db4xDtj7aOfzAX4aZy1/view?usp=share_link)
- one from [Indy multisession RTT](https://wandb.ai/joelye9/context_general_bci/runs/uych1wae?workspace=user-joelye9): [Checkpoint](https://drive.google.com/file/d/1hhC4n1UyiYjCcv1nlO6ESljNhr8qVlUF/view?usp=share_link).

## Other Notes
- The codebase was actually developed in Python 3.10 but this release uses 3.9 for compatibility with `onnx`. Exact numerical reproduction of paper results is not asserted, but please file an issue if large discrepancies with reported results arise.
- Check out ./scripts/figures/` and this [wandb workspace](https://wandb.ai/joelye9/context_general_bci) to see how the results were generated.




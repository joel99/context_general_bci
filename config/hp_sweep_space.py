# For sampling with utils.generate_search

# Really the majority of experiments are about _data_
# so `dataset` (or embedding strat) is the configured item, and we sweep other nuisance params.
# Formatted according to expectations in `halton.py`

sweep_space = {
    "ft_reg": {
        "model.dropout": {
            'feasible_points': [0.1, 0.3, 0.5, 0.7]
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        }
    },
    "lr": {
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            'scaling': 'log',
        },
        "model.lr_ramp_steps": {
            'feasible_points': [10, 25, 50, 100],
        },
    },
    "lr_and_dropout": {
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            'scaling': 'log',
        },
        "model.lr_ramp_steps": {
            'feasible_points': [10, 25, 50, 100],
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.3, 0.5, 0.7] # in lieu of sweeping capacity
        }
    },
    'lr_v3': {
        "model.lr_init": {
            'min': 2e-4,
            'max': 8e-4,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.05, 0.1, 0.2]
        }
    },
    "lr_v2": {
        "model.lr_init": {
            'min': 2e-4,
            # 'min': 1e-4,
            'max': 1e-3,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.3, 0.5, 0.7] # in lieu of sweeping capacity
        }
    }, # post-mortem. dropout of 0.7 kills unless carefully regulated, don't do this. Sweep hidden size instead.
    "base": {
        # we will use a fixed 6-layer architecture for now, sweep hidden.
        "model.hidden_size": {
            'feasible_points': [128, 256, 384]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            # 'max': 5e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "large": {
        "model.hidden_size": {
            'feasible_points': [256, 384, 512, 768]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 2e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "small_wide": {
        # we will use a fixed 6-layer architecture for now, sweep hidden.
        "model.hidden_size": {
            'feasible_points': [128, 192, 256]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            # 'max': 5e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
}

r"""
- Noted fixed parameters
    - lr schedule (cosine decay) and decay step (interminably long horizon)
    - adam hyperparams besides lr (despite playbook recommendation) - we don't have budget
"""
# For sampling with utils.generate_search

# Really the majority of experiments are about _data_
# so `dataset` (or embedding strat) is the configured item, and we sweep other nuisance params.
# Formatted according to expectations in `halton.py`

sweep_space = {
    "base": {
        # we will use a fixed 6-layer architecture for now, sweep hidden.
        "model.hidden_size": {
            'feasible_points': [128, 256, 384]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 2e-2, # models reliably crash past 1e-2
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
            'max': 5e-3, # models reliably crash past 1e-2
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
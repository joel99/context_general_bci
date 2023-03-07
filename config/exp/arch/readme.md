# Fig 1: Architecture superiority

Experiments for architecture comparisons. Derived from `saturation` line of exps.
Diffs:
- we use same test session as in data sourcing transfer experiments for consistency. (we picked `Indy-20160627_01` there for having a large number of trials and being relatively centered in time).
- control number of test session trials - keeping it relatively small (but still practically a functionally relevant "large calibration" regime) - 5 minutes; so as to amplify sample efficiency differences.

`base`:
Causal, Sorted, Test Session restricted to 300 trials (5 minutes). ~26K trials.
- Patch 4, 16, 32, 128
- Stitch
- NDT-1 (Time), 6 and 8 layer (control for capacity).

Batch size controlled at 256, though we don't expect it to matter much.

`scaledown`:
TODO...
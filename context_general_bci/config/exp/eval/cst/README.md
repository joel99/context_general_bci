Currently these experiments NaN out. Unable to trace the path of the NaN due to time constraints. Known symptoms:
Working backward from the NaN:
- Doesn't appear to come from a NaN loss
- NaN is inconsistently caught in on_train_batch_end
- NaN is inconsistently caught when full model weights NaN out
Working forward from construction:
- NaN occurs at a random epoch, noted from epoch 2 to 100. We are clipping gradients; clipping it to tiny values does not  help.
- Data is square, so there's no reason to suspect.
- Disabling either spike_infill or kinematic decoding causes the NaN to go away. This obscures the path and suggests it's independent of input batch.
- Changing seed changes NaN location.

Because we were unable to pin down the source of NaN in standard training, we disable the neural reconstruction loss.
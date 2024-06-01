#! /usr/bin/env python

import pytorch_lightning as pl
import torch
import yaml

from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score

class RNN(pl.LightningModule):
    """ Class to create Recurrent Neural Networks (RNNs) using Pytorch 
    Lightning for BrainGate2 experiments.
    """

    def __init__(self, train=True, input_dim=None):
        """ Initializes RNN architecture and parameters.

        Parameters
        ----------
        train : bool, Default: True
            Specifies whether model instantiation is for training or
            real-time implementation. Necessary for predetermination
            of paths and storing hyperparameters.
        """

        super().__init__()

        # Define path to training YAML file
        # Depends on whether training model or not
        if train == True:
            train_yaml = 'src/train_RNN.yaml'
        else:
            train_yaml = '../brand-modules/brand-emory/nodes/RNN_decoder/src/train_RNN.yaml'

        # Load hyperparameters from YAML file
        with open(train_yaml, 'r') as file:
            self.params = yaml.safe_load(file)

        # Save hyperparameters used to train model
        if train == True:
            self.save_hyperparameters(self.params)

        # ------------ Define model architecture --------------- #
        input_size = (input_dim
                      if input_dim else self.params['model_dim']['input_dim'])
        self.LSTM = nn.LSTM(input_size=input_size,
                            hidden_size=self.params['model_dim']['hidden_dim'],
                            num_layers=self.params['model_dim']['n_layers'],
                            batch_first=True)

        self.fc = nn.Linear(
            in_features = self.params['model_dim']['hidden_dim'],
            out_features = self.params['model_dim']['output_dim']
        )

        self.Dropout = nn.Dropout(
            p=self.params['model_hparams']['dropout']
        )

    def forward(self, input):
        """ Forward pass of RNN model.

        Parameters
        ----------
        input: torch.tensor
            Tensor of binned spiking data.
            Shape: (batch_size, seq_len, n_features)

        Returns
        -------
        out: torch.tensor
            Target predictions.
            Shape: ()
        """

        # Dropout the input
        input = self.Dropout(input)

        # Dropped out input through LSTM layer(s)
        output, (hn, cn) = self.LSTM(input)

        # return preds from end of sequences
        out = self.fc(output[:, -1, :])

        return out

    def configure_optimizers(self):
        """ Declares Optimizer and Scheduler
        Objects used during training.

        Returns
        -------
        opt_dict: dict
            Dictionary containing the optimizer,
            Learn Rate scheduler, and value to
            monitor for LR scheduling.
        """

        # Create Optimizer Object
        adam_opt = torch.optim.Adam(
            self.parameters(),
            lr=self.params['model_hparams']['learn_rate'],
            weight_decay= self.params['model_hparams']['weight_decay']
        )

        # Create LR Scheduler object
        scheduler = ReduceLROnPlateau(
            optimizer=adam_opt,
            mode='min',
            patience=self.params['callbacks']['scheduler_patience'],
            factor=0.1,
            threshold=0.001
        )

        opt_dict = {
            'optimizer': adam_opt,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

        return opt_dict

    def MSE_loss(self, preds, labels):
        """ Define loss function to use
        during training.

        Parameters
        ----------
        preds: torch.tensor
            Target Predictions from RNN.

        labels: torch.tensor
            True target values.

        Returns
        -------
        MSE loss value.
        """
        return F.mse_loss(preds, labels)

    def training_step(self, train_batch, batch_idx):
        """ Defines Training Loop for each epoch.

        Parameters
        ----------
        train_batch: tuple
            Tuple holding input and target data.

        batch_idx: Ignore

        Returns
        -------
        train_loss: float
            Loss calculated from the training
            predictions and true targets.
        """

        #Extract input, target data from DataLoader
        spikes, true_vels = train_batch

        # Check if Gaussian Noise is added to spikes
        if self.params['model_hparams']['gauss_noise'] == True:
            noise = torch.normal(0, 0.3, size=spikes.size(), device=self.device)
            spikes = spikes + noise

        # Inference
        vel_preds = self.forward(spikes)

        vels_true = true_vels[:, -1, :]

        # Calculate Loss
        train_loss = self.MSE_loss(vel_preds, vels_true)

        # Calculate r^2 value
        train_r2 = r2_score(
            vels_true.detach().cpu().numpy(), #add .squeeze
            vel_preds.detach().cpu().numpy(),
        )

        training_dict = {
            'train_loss': train_loss,
            'training R^2': train_r2
        }

        # Log Training loss and r^2
        self.log_dict(
            training_dict,
            on_step = False,
            on_epoch=True,
            prog_bar=True
        )

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        """ Defines Validation Loop for each epoch

        Parameters
        ----------
        val_batch: tuple
            Tuple holding input and target data.

        batch_idx: Ignore

        Returns
        -------
        val_loss: float
            Loss calculated from the validation
            predictions and true targets.
        """

        #Extract input, target data from DataLoader
        spikes, true_vels = val_batch

        # Inference
        vel_preds = self.forward(spikes)

        vels_true = true_vels[:, -1, :]

        # Calculate Loss
        val_loss = self.MSE_loss(vel_preds, vels_true)

        # Calculate r^2 value
        val_r2 = r2_score(
            vels_true.detach().cpu().numpy(), #add .squeeze
            vel_preds.detach().cpu().numpy()
        )

        val_dict = {
            'val_loss': val_loss,
            'val R^2': val_r2
        }

        # Log validation loss and r^2
        self.log_dict(
            val_dict,
            on_step = False,
            on_epoch=True,
            prog_bar=True
        )

        return val_loss
import os

import torch
from sklearn.model_selection import train_test_split
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Settings
from utils import get_train_transform, get_test_transform
from dataset import BarcodeDataset
from utils import set_deterministic, get_train_transform, get_test_transform, Mode, visualize
from loss import DetectionClassificationLoss
import shutil

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TrainingModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_func = DetectionClassificationLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)
        total_loss = self.loss_func(y_pred, y)

        metrics = {'train_total_loss': total_loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)
        total_loss = self.loss_func(y_pred, y)

        metrics = {'val_total_loss': total_loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=False)

        visualize(x, y, y_pred, batch_idx)

        return total_loss

    def predict_step(self, batch, batch_idx):
        thresh = 0.5
        x = batch
        y_bin_logit, y_class_logit = self.model(x)
        bin_y = torch.where(F.sigmoid(y_bin_logit) > thresh, 1, 0)
        class_y = F.softmax(y_class_logit, dim=1).argmax(dim=1)
        return bin_y, class_y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=Settings.lr, weight_decay=Settings.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_total_loss',
                'frequency': 1,
            },
        }


def train_model(model):
    if Settings.seed is not None:
        set_deterministic(Settings.seed)

    filenames = sorted(os.listdir(Settings.dataset_path + '/IMAGES'))
    train_filenames, val_filenames = train_test_split(filenames, train_size=0.8, shuffle=True,
                                                      random_state=Settings.seed)
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    train_dataset = BarcodeDataset(Mode.TRAIN, Settings.dataset_path + '/IMAGES', Settings.dataset_path + '/MASKS',
                                   train_filenames, train_transform)
    val_dataset = BarcodeDataset(Mode.VAL, Settings.dataset_path + '/IMAGES', Settings.dataset_path + '/MASKS',
                                 val_filenames, test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=Settings.batch_size,
                                  shuffle=True, num_workers=Settings.num_workers,
                                  pin_memory=Settings.pin_memory, persistent_workers=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=Settings.batch_size,
                                       shuffle=False, num_workers=Settings.num_workers,
                                       pin_memory=Settings.pin_memory, persistent_workers=True)

    Settings.train_steps_per_epoch = len(train_dataloader)

    earlystopping_callback = EarlyStopping(
        monitor='val_total_loss',
        mode='min',
        patience=Settings.early_stopping_patience,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'runs/{Settings.model_name}',
        filename='{epoch}-{val_acc:.3f}',
        monitor='val_total_loss',
        mode='min',
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=Settings.num_epochs,
        accelerator=Settings.accelerator,
        devices=1,
        callbacks=[earlystopping_callback, checkpoint_callback],
        log_every_n_steps=Settings.log_every_n_steps,
        enable_checkpointing=True,
        logger=True,
    )

    training_module = TrainingModule(model=model)
    trainer.fit(training_module, train_dataloader, validation_dataloader)

    shutil.copy(checkpoint_callback.best_model_path, Settings.model_path)

    return training_module.model

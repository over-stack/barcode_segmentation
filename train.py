import os

import torch
from sklearn.model_selection import train_test_split
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader

import config
from config import Settings
from utils import get_train_transform, get_test_transform
from dataset import BarcodeDataset
from utils import (set_deterministic, get_train_transform, get_test_transform, Mode, visualize_input, visualize_output,
                   postporcess_base, postprocess_embeddings, postprocess_objects, visualize_embeddings)
from loss import DetectionClassificationLoss, DiscriminativeLoss, calculate_means
import shutil

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TrainingModule(pl.LightningModule):
    def __init__(self, model, base_loss_weight, embedding_loss_weight):
        super().__init__()
        self.model = model
        self.base_loss_weight = base_loss_weight
        self.embedding_loss_weight = embedding_loss_weight

        self.base_loss_func = DetectionClassificationLoss()
        self.embedding_loss_func = DiscriminativeLoss(config.Settings.delta_var, config.Settings.delta_dist,
                                                      config.Settings.norm)

        self.total_means_sum: torch.Tensor | None = None  # !!! set zero after epoch
        self.total_means_count: int = 0  # !!! set zero after epoch
        self.on_validation = False

        torch.set_float32_matmul_precision('medium')

    def training_step(self, batch, batch_idx):
        x, y, n_objects = batch

        if self.on_validation:
            self.total_means_sum = None
            self.total_means_count = 0
            self.on_validation = False

        base_pred, embeddings_pred = self.model(x)

        base_loss = self.base_loss_func(base_pred, y)
        embedding_loss = self.embedding_loss_func(embeddings_pred, y, n_objects)
        total_loss = self.base_loss_weight * base_loss + self.embedding_loss_weight * embedding_loss

        metrics = {'train_total_loss': total_loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=False)

        with torch.no_grad():
            batch_means = calculate_means(embeddings_pred, y, n_objects, not_from_loss=True).mean(dim=0)
            if self.total_means_sum is None:
                self.total_means_sum = torch.zeros(batch_means.shape, device=self.device)
                # print(batch_means.shape)
            self.total_means_sum += batch_means
            self.total_means_count += 1

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, n_objects = batch

        if not self.on_validation:
            self.on_validation = True

        base_pred, embeddings_pred = self.model(x)

        base_loss = self.base_loss_func(base_pred, y)
        embedding_loss = self.embedding_loss_func(embeddings_pred, y, n_objects)
        total_loss = self.base_loss_weight * base_loss + self.embedding_loss_weight * embedding_loss

        metrics = {'val_total_loss': total_loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=False)

        # batch_idx % 4 == 0
        if True and self.total_means_sum is not None:
            with torch.no_grad():
                means = self.total_means_sum / self.total_means_count
                base_pred_post = postporcess_base(base_pred)
                embeddings_pred_post = postprocess_embeddings(embeddings_pred, means,
                                                              n_instances=y.shape[1], bandwidth=Settings.delta_var,
                                                              norm=Settings.norm)
                visualize_input(x, batch_idx, 'input')
                visualize_output(y, batch_idx, 'output')
                visualize_output(base_pred_post, batch_idx, 'pred_base')
                visualize_output(postprocess_objects(base_pred_post, min_object_area=20),
                                 batch_idx, 'pred_base_objs')
                visualize_output(embeddings_pred_post, batch_idx, 'pred_embed')
                visualize_output(postprocess_objects(embeddings_pred_post, min_object_area=20),
                                 batch_idx, 'pred_embed_objs')
                visualize_embeddings(embeddings_pred, embeddings_pred_post, means, batch_idx, 'viz_embed')

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
            patience=2,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_total_loss',
                'frequency': 1,
            },
        }


def train_model(model, base_loss_weight, embedding_loss_weight):
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

    training_module = TrainingModule(model, base_loss_weight, embedding_loss_weight)
    trainer.fit(training_module, train_dataloader, validation_dataloader)

    shutil.copy(checkpoint_callback.best_model_path, Settings.model_path)

    return training_module.model

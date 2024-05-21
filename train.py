import os
import time

import torch
from sklearn.model_selection import train_test_split
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader

import config
from config import Settings
from utils import get_train_transform, get_test_transform, seed_worker
from dataset import BarcodeDataset, BarcodeDatasetXML
from utils import (set_deterministic, get_train_transform, get_test_transform, Mode,
                   postporcess_base, postprocess_embeddings, postprocess_objects)
from visualization import Visualization
from metrics import TotalMetrics
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

        self.base_loss_func = DetectionClassificationLoss(
            Settings.weight_positive, Settings.weight_negative, Settings.weight_k_worst_negative,
            Settings.detection_loss_weight, Settings.classification_loss_weight, None
        )
        self.embedding_loss_func = DiscriminativeLoss(
            Settings.delta_var, Settings.delta_dist, Settings.norm,
            Settings.var_term_weight, Settings.dist_term_weight, Settings.reg_term_weight, None
        )
        self.visualization = Visualization(n_cols=Settings.n_cols, n_rows=Settings.n_rows)

        self.train_metrics_base = TotalMetrics(iou_thresholds=Settings.metrics_thresholds)
        self.train_metrics_embed = TotalMetrics(iou_thresholds=Settings.metrics_thresholds)
        self.validation_metrics_base = TotalMetrics(iou_thresholds=Settings.metrics_thresholds)
        self.validation_metrics_embed = TotalMetrics(iou_thresholds=Settings.metrics_thresholds)

        # for means calculation (on last train epoch, before grad step), using on eval epoch
        self.total_means_sum: torch.Tensor | None = None  # !!! set zero after epoch
        self.total_batches_count: int = 0  # !!! set zero after epoch
        self.on_validation = False  # validation is running

        self.sum_mean_inference_time_base = 0.0
        self.sum_mean_inference_time_embed = 0.0
        self.validation_steps_count = 0

        torch.set_float32_matmul_precision('medium')

    def training_step(self, batch, batch_idx):
        x, y, n_objects = batch

        if self.on_validation:
            self.total_means_sum = None
            self.total_batches_count = 0
            self.on_validation = False

        base_pred, embeddings_pred = self.model(x)

        base_loss = self.base_loss_func(base_pred, y)
        embedding_loss = self.embedding_loss_func(embeddings_pred, y, n_objects)
        total_loss = self.base_loss_weight * base_loss + self.embedding_loss_weight * embedding_loss

        with torch.no_grad():
            '''base_pred_post = postporcess_base(base_pred)
            pred_masks_base, pred_boxes_base, pred_labels_base = postprocess_objects(
                base_pred_post, min_object_area=Settings.min_object_area
            )
            _, target_boxes, target_labels = postprocess_objects(
                y, min_object_area=Settings.min_object_area
            )

            self.train_metrics.update(
                y[:, 1:], target_boxes, target_labels, pred_masks_base[:, 1:], pred_boxes_base, pred_labels_base
            )'''

            batch_means = calculate_means(embeddings_pred, y, n_objects, not_from_loss=True).mean(dim=0)
            if self.total_means_sum is None:
                self.total_means_sum = torch.zeros(batch_means.shape, device=self.device)
            self.total_means_sum += batch_means
            self.total_batches_count += 1

        metrics = {'train_total_loss': total_loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, n_objects = batch

        if not self.on_validation:
            self.on_validation = True

        base_pred, embeddings_pred = self.model(x)

        base_loss = self.base_loss_func(base_pred, y)
        embedding_loss = self.embedding_loss_func(embeddings_pred, y, n_objects)
        total_loss = self.base_loss_weight * base_loss + self.embedding_loss_weight * embedding_loss

        with torch.no_grad():
            self.validation_steps_count += 1
            base_inference_time = time.time()
            base_pred_post = postporcess_base(base_pred, Settings.postprocessing_threshold)
            base_inference_time = 1000 * (time.time() - base_inference_time) / x.shape[0]
            self.sum_mean_inference_time_base += base_inference_time

            pred_masks_base, pred_boxes_base, pred_labels_base = postprocess_objects(
                base_pred_post, min_object_area=Settings.min_object_area
            )
            _, target_boxes, target_labels = postprocess_objects(
                y, min_object_area=Settings.min_object_area
            )
            self.validation_metrics_base.update(
                y[:, 1:], target_boxes, target_labels, pred_masks_base[:, 1:], pred_boxes_base, pred_labels_base
            )

            if self.total_means_sum is not None and True:
                means = self.total_means_sum / self.total_batches_count
                embed_inference_time = time.time()
                embeddings_pred_post = postprocess_embeddings(
                    embeddings_pred, means, n_instances=y.shape[1], bandwidth=Settings.delta_var, norm=Settings.norm
                )
                embed_inference_time = 1000 * (time.time() - embed_inference_time) / x.shape[0]
                self.sum_mean_inference_time_embed += embed_inference_time
                pred_masks_embed, pred_boxes_embed, pred_labels_embed = postprocess_objects(
                    embeddings_pred_post, min_object_area=20
                )
                self.validation_metrics_embed.update(
                    y[:, 1:], target_boxes, target_labels, pred_masks_embed[:, 1:], pred_boxes_embed, pred_labels_embed
                )

                self.visualization.visualize(
                    x, y, batch_idx, base_pred_post, pred_masks_base, embeddings_pred_post, pred_masks_embed,
                    embeddings_pred, means
                )

        metrics = {'val_total_loss': total_loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=False)

        return total_loss

    def on_validation_epoch_end(self):
        epoch_metrics_base = self.validation_metrics_base.compute()
        epoch_metrics_embed = self.validation_metrics_embed.compute()
        print()
        print('Base:', epoch_metrics_base)
        print('Embed:', epoch_metrics_embed)
        print(f'Base inference time: {self.sum_mean_inference_time_base / self.validation_steps_count :.6f}')
        print(f'Embed inference time: {self.sum_mean_inference_time_embed / self.validation_steps_count :.6f}')
        print()
        self.validation_metrics_base.reset()
        self.validation_metrics_embed.reset()
        self.sum_mean_inference_time_base = 0
        self.sum_mean_inference_time_embed = 0
        self.validation_steps_count = 0

    def predict_step(self, batch, batch_idx):
        # TODO: update this function
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
            factor=Settings.lr_reduce_factor,
            patience=Settings.lr_reduce_patience,
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

    filenames = sorted(os.listdir(Settings.dataset_path + '/Image'))
    # filenames = sorted(os.listdir(Settings.dataset_path + '/IMAGES'))
    train_filenames, val_filenames = train_test_split(filenames, train_size=0.8, shuffle=True,
                                                      random_state=Settings.seed)
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    train_dataset = BarcodeDatasetXML(Mode.TRAIN, Settings.dataset_path + '/Image', Settings.dataset_path + '/Markup',
                                      train_filenames, train_transform)
    val_dataset = BarcodeDatasetXML(Mode.VAL, Settings.dataset_path + '/Image', Settings.dataset_path + '/Markup',
                                    val_filenames, test_transform)
    '''train_dataset = BarcodeDataset(Mode.TRAIN, Settings.dataset_path + '/IMAGES', Settings.dataset_path + '/MASKS',
                                   train_filenames, train_transform)
    val_dataset = BarcodeDataset(Mode.VAL, Settings.dataset_path + '/IMAGES', Settings.dataset_path + '/MASKS',
                                 val_filenames, test_transform)'''
    g = torch.Generator()
    g.manual_seed(Settings.seed)

    train_dataloader = DataLoader(
        train_dataset, batch_size=Settings.batch_size, shuffle=True, num_workers=Settings.num_workers,
        pin_memory=Settings.pin_memory, persistent_workers=True, worker_init_fn=seed_worker, generator=g
    )
    validation_dataloader = DataLoader(
        val_dataset, batch_size=Settings.batch_size, shuffle=False, num_workers=Settings.num_workers,
        pin_memory=Settings.pin_memory, persistent_workers=True, worker_init_fn=seed_worker, generator=g
    )

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

import torch
import torchvision
import random
import numpy as np
import os
import albumentations as A
from PIL import Image

from config import Settings
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import functional as AF
from torch.nn import functional as F
from enum import Enum
import matplotlib.pyplot as plt

from torchvision.transforms import Resize
import cv2


class Visualization:
    def __init__(self, n_cols=5, n_rows=2):
        self.n_cols = n_cols
        self.n_rows = n_rows

    def visualize(self, x, y, batch_idx, base_pred_post, result_masks_base, embeddings_pred_post, result_masks_embed,
                  embeddings_pred, means):
        self.visualize_input(x, batch_idx, 'input')
        self.visualize_output(y, batch_idx, 'output')
        self.visualize_output(base_pred_post, batch_idx, 'pred_base')
        self.visualize_output(result_masks_base, batch_idx, 'pred_base_objs')
        self.visualize_output(embeddings_pred_post, batch_idx, 'pred_embed')
        self.visualize_output(result_masks_embed, batch_idx, 'pred_embed_objs')
        self.visualize_embeddings(embeddings_pred, embeddings_pred_post, means, batch_idx, 'viz_embed')

    def visualize_input(self, x: torch.Tensor, batch_idx: int, filename: str):
        x = x.to('cpu')
        n_images = self.n_cols * self.n_rows
        bs, n_channels, height, width = x.shape
        torch_image_mean = torch.tensor(Settings.mean).unsqueeze(0).expand(n_images, n_channels).unsqueeze(2).unsqueeze(3)
        torch_image_std = torch.tensor(Settings.std).unsqueeze(0).expand(n_images, n_channels).unsqueeze(2).unsqueeze(3)
        grid = torchvision.utils.make_grid(255 * (x[:n_images] * torch_image_std + torch_image_mean), nrow=self.n_cols)
        grid_numpy = grid.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        im = Image.fromarray(grid_numpy)
        im.save(f'images/{batch_idx}_{filename}.png')

    def visualize_output(self, y: torch.Tensor, batch_idx: int, filename: str, first_white=False):
        y = y.to('cpu')
        n_images = self.n_cols * self.n_rows
        bs, n_classes, height, width = y.shape
        # (n_images, n_classes, height, width) * (n_images, [n_classes, 3], height, width) =>
        # => (n_images, 3, height, width)
        class_colors = Settings.class_colors[:n_classes]
        if first_white:
            class_colors = [torch.tensor([255, 255, 255])] + class_colors[:-1]
        colorize_matrix = torch.stack(class_colors, dim=0)
        _y = y[:n_images].permute(0, 2, 3, 1).unsqueeze(3)
        colorize_matrix = colorize_matrix.reshape(1, 1, 1, n_classes, 3).expand(n_images, height, width, n_classes, 3)
        y_colored = (_y @ colorize_matrix).squeeze(3).permute(0, 3, 1, 2)

        resize_transform = Resize((y.shape[2] * Settings.output_downscale, y.shape[3] * Settings.output_downscale),
                                  interpolation=torchvision.transforms.InterpolationMode('nearest'))
        grid = torchvision.utils.make_grid(resize_transform(y_colored), nrow=self.n_cols)
        grid_numpy = grid.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        im = Image.fromarray(grid_numpy)
        im.save(f'images/{batch_idx}_{filename}.png')

    def visualize_embeddings(self, y_pred: torch.Tensor, y_pred_post: torch.Tensor, means: torch.Tensor,
                             batch_idx: int, filename: str):
        """
        :param y_pred: (bs, n_dims, height, width)
        :param y_pred_post: (bs, n_instances, height, width)
        :param means: (n_instances, n_dims)
        :param batch_idx:
        :param filename:
        :param n_cols:
        :param n_rows:
        :return:
        """

        # TODO: draw means

        _, n_dims, height, width = y_pred.shape
        assert n_dims == 2, 'Visualization is available only for 2D embeddings'
        _, n_instances, _, _ = y_pred_post.shape
        n_images = self.n_cols * self.n_rows

        # points
        y_pred = y_pred[:n_images].permute(0, 2, 3, 1).reshape(n_images, -1, n_dims)
        # colors
        y_pred_post = y_pred_post[:n_images].argmax(dim=1).unsqueeze(-1).reshape(n_images, -1, 1)

        y_range = max(abs(torch.max(y_pred[:, :, 0])), abs(torch.min(y_pred[:, :, 0])))
        x_range = max(abs(torch.max(y_pred[:, :, 1])), abs(torch.min(y_pred[:, :, 1])))

        spacing = 10  # >= 1
        y_pred[:, :, 0] = torch.round(y_pred[:, :, 0] / y_range * (height - spacing) / 2 + (height - spacing) / 2)
        y_pred[:, :, 1] = torch.round(y_pred[:, :, 1] / x_range * (width - spacing) / 2 + (width - spacing) / 2)
        y_pred = y_pred.to(torch.int64)

        result = list()
        for image_idx in range(n_images):
            result_image = torch.zeros((height, width), device=Settings.device, dtype=torch.int64)
            indices = y_pred[image_idx].unsqueeze(1)
            color_indices = y_pred_post[image_idx]
            for color_idx in range(n_instances):
                color_mask_coord = indices[color_indices == color_idx]
                # + 1 because 0 (black) is already taken as class
                result_image[color_mask_coord[:, 0], color_mask_coord[:, 1]] = color_idx + 1
            result.append(result_image)
        result = torch.stack(result, dim=0).to(y_pred.device)
        result = F.one_hot(result, num_classes=n_instances + 1).permute(0, 3, 1, 2)
        self.visualize_output(result, batch_idx, filename, first_white=True)

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


class Mode(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_train_transform():
    transform = A.Compose([
        A.LongestMaxSize(max_size=Settings.width),
        A.PadIfNeeded(min_height=Settings.height, min_width=Settings.width,
                      position=A.PadIfNeeded.PositionType.TOP_LEFT, value=(155, 155, 155)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=20, p=0.85),
        A.Normalize(mean=Settings.mean, std=Settings.std, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transform


def get_test_transform():
    transform = A.Compose([
        A.LongestMaxSize(max_size=Settings.width),
        A.PadIfNeeded(min_height=Settings.height, min_width=Settings.width,
                      position=A.PadIfNeeded.PositionType.TOP_LEFT, value=(155, 155, 155)),
        A.Normalize(mean=Settings.mean, std=Settings.std, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transform


def visualize_input(x: torch.Tensor, batch_idx: int, filename: str, n_cols: int = 5, n_rows: int = 2):
    x = x.to('cpu')
    n_images = n_cols * n_rows
    bs, n_channels, height, width = x.shape
    torch_image_mean = torch.tensor(Settings.mean).unsqueeze(0).expand(n_images, n_channels).unsqueeze(2).unsqueeze(3)
    torch_image_std = torch.tensor(Settings.std).unsqueeze(0).expand(n_images, n_channels).unsqueeze(2).unsqueeze(3)
    grid = torchvision.utils.make_grid(255 * (x[:n_images] * torch_image_std + torch_image_mean), nrow=n_cols)
    grid_numpy = grid.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    im = Image.fromarray(grid_numpy)
    im.save(f'images/{batch_idx}_{filename}.png')


def visualize_output(y: torch.Tensor, batch_idx: int, filename: str, n_cols: int = 5, n_rows: int = 2, first_white=False):
    y = y.to('cpu')
    n_images = n_cols * n_rows
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
    grid = torchvision.utils.make_grid(resize_transform(y_colored), nrow=n_cols)
    grid_numpy = grid.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    im = Image.fromarray(grid_numpy)
    im.save(f'images/{batch_idx}_{filename}.png')


def visualize_embeddings(y_pred: torch.Tensor, y_pred_post: torch.Tensor, means: torch.Tensor,
                         batch_idx: int, filename: str, n_cols: int = 5, n_rows: int = 2):
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
    n_images = n_cols * n_rows

    # points
    y_pred = y_pred[:n_images].permute(0, 2, 3, 1).reshape(n_images, -1, n_dims)
    # colors
    y_pred_post = y_pred_post[:n_images].argmax(dim=1).unsqueeze(-1).reshape(n_images, -1, 1)

    y_range = max(abs(torch.max(y_pred[:, :, 0])), abs(torch.min(y_pred[:, :, 0])))
    x_range = max(abs(torch.max(y_pred[:, :, 1])), abs(torch.min(y_pred[:, :, 1])))

    spacing = 10  # >= 1
    y_pred[:, :, 0] = torch.round(y_pred[:, :, 0] / (2 * y_range) * (height - spacing) / 2 + (height + spacing) / 2)
    y_pred[:, :, 1] = torch.round(y_pred[:, :, 1] / (2 * x_range) * (width - spacing) / 2 + (width + spacing) / 2)
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
    result = F.one_hot(result).permute(0, 3, 1, 2)
    visualize_output(result, batch_idx, filename, n_cols, n_rows, first_white=True)


def postprocess_objects(prediction: torch.Tensor, min_object_area: int = 20):
    prediction = prediction.detach().to('cpu')
    objects_masks = (prediction[:, 0] != 1)
    classes = prediction.argmax(dim=1).numpy()

    result = list()
    for image_idx in range(prediction.shape[0]):
        contours, _ = cv2.findContours(np.array(objects_masks[image_idx], dtype=np.uint8), mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda contour: cv2.contourArea(contour) > min_object_area, contours))
        rectangles = [cv2.minAreaRect(contour) for contour in contours]
        boxes = [np.int0(cv2.boxPoints(rect)) for rect in rectangles]

        empty_mask = np.zeros(objects_masks[image_idx].shape, dtype=np.uint8)
        image_result = np.zeros(objects_masks[image_idx].shape, dtype=np.uint8)
        for i in range(len(boxes)):
            object_mask = cv2.drawContours(empty_mask, boxes, i, color=1, thickness=-1)
            object_class = round(np.mean(object_mask * classes[image_idx]))
            image_result[object_mask != 0] = object_class
        result.append(image_result)

    result = torch.from_numpy(np.stack(result, axis=0)).to(torch.int64).to(prediction.device)  # cpu
    result = F.one_hot(result).permute(0, 3, 1, 2)
    return result


def postporcess_base(prediction: torch.Tensor, prob_threshold: float = 0.5):
    """
    :param prediction: (bs, 1 + n_classes, height, width)
    :param prob_threshold: float[0, 1]
    :return: (bs, 1 + n_classes, height, width)
    """

    bin_logits = prediction[:, 0]
    logit_threshold = - np.log(1 / np.clip(prob_threshold, Settings.eps, 1 - Settings.eps) - 1)
    bin_logits = torch.where(bin_logits > logit_threshold, 1, 0)

    classes = F.softmax(prediction[:, 1:], dim=1).argmax(dim=1) + 1
    result = (classes * bin_logits).to(torch.int64)
    result = F.one_hot(result).permute(0, 3, 1, 2)

    # bs, n_instances, height, width
    return result


def postprocess_embeddings(prediction: torch.Tensor, means: torch.Tensor, n_instances: int, bandwidth: float, norm: int):
    """
    :param prediction: (bs, n_dims, height, width)
    :param means: (n_instances, n_dims)
    :param n_instances: int
    :param bandwidth: float
    :param norm: int
    :return: (bs, n_instances, height, width)
    """

    bs, n_dims, height, width = prediction.shape
    n_loc = height * width
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(bs, n_loc, n_dims)
    prediction = prediction.unsqueeze(2).expand(bs, n_loc, n_instances, n_dims)
    means = means.unsqueeze(0).expand(n_loc, n_instances, n_dims)
    means = means.unsqueeze(0).expand(bs, n_loc, n_instances, n_dims)

    eps = 1e-5
    # bs, n_loc, n_instances
    result = torch.clamp(torch.norm((prediction - means), norm, 3) - bandwidth, min=0.0)
    result = (result < eps).reshape(bs, height, width, n_instances).permute(0, 3, 1, 2)

    # bs, n_instances, height, width
    return result.to(torch.int64)

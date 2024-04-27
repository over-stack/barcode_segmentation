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
        A.LongestMaxSize(max_size=Settings.height),
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
        A.LongestMaxSize(max_size=Settings.height),
        A.PadIfNeeded(min_height=Settings.height, min_width=Settings.width,
                      position=A.PadIfNeeded.PositionType.TOP_LEFT, value=(155, 155, 155)),
        A.Normalize(mean=Settings.mean, std=Settings.std, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transform


def resize(image, mask, image_max_side, image_side_multiple):
    assert len(image.shape) <= 3, "Alpha channel error"
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    if max(height, width) > image_max_side:
        scale_factor = image_max_side / max(height, width)
        if height > width:
            new_height = image_max_side
            new_width = max(1, round(width * scale_factor / image_side_multiple)) * image_side_multiple
        else:
            new_height = max(1, round(height * scale_factor / image_side_multiple)) * image_side_multiple
            new_width = image_max_side
    else:
        new_width = max(1, round(width / image_side_multiple)) * image_side_multiple
        new_height = max(1, round(height / image_side_multiple)) * image_side_multiple

    resized_image = image.resize(size=(new_width, new_height), resample=Image.BICUBIC)
    resized_mask = None
    if mask:
        resized_mask = mask.resize(size=(new_width, new_height), resample=Image.NEAREST)

    return resized_image, resized_mask


def extract_contours_and_boxes(bin_logits: torch.Tensor, prob_threshold=0.5, min_area=10):
    logit_threshold = - np.log(1 / np.clip(prob_threshold, Settings.eps, 1 - Settings.eps) - 1)
    bin_logits = torch.where(bin_logits > logit_threshold, 1, 0)
    contours, _ = cv2.findContours(np.array(bin_logits, dtype=np.uint8),
                                   mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda contour: cv2.contourArea(contour) > min_area, contours))
    rectangles = [cv2.minAreaRect(contour) for contour in contours]
    boxes = [np.int0(cv2.boxPoints(rect)) for rect in rectangles]

    mask = np.zeros(bin_logits.shape, dtype=np.uint8)
    mask = cv2.drawContours(mask, boxes, -1, color=255, thickness=-1)

    return mask


def visualize_embeddings(y_pred: torch.Tensor, y_pred_post: torch.Tensor, batch_idx: int,
                         n_cols: int = 5, n_rows: int = 2):
    n_images = n_cols * n_rows
    class_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    y_pred = y_pred[:n_images].permute(0, 2, 3, 1).to('cpu').numpy()
    y_pred_post = y_pred_post[:n_images].argmax(dim=1).to('cpu').numpy()

    # plt.figure()
    # plt.plot()


def visualize(x: torch.Tensor | None, y: torch.Tensor | None, y_pred: torch.Tensor | None,
              batch_idx: int, n_cols: int = 5, n_rows: int = 2):

    n_images = n_cols * n_rows
    class_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    class_colors = list([torch.tensor(color) for color in class_colors])

    if x is not None:
        x = x.to('cpu')
        bs, n_channels, height, width = x.shape
        torch_image_mean = torch.tensor(Settings.mean).unsqueeze(0).expand(n_images, n_channels).unsqueeze(2).unsqueeze(3)
        torch_image_std = torch.tensor(Settings.std).unsqueeze(0).expand(n_images, n_channels).unsqueeze(2).unsqueeze(3)
        grid = torchvision.utils.make_grid(255 * (x[:n_images] * torch_image_std + torch_image_mean), nrow=n_cols)
        grid_numpy = grid.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        im = Image.fromarray(grid_numpy)
        im.save(f'images/{batch_idx}_input.png')

    if y is not None:
        y = y.to('cpu')
        bs, n_classes, height, width = y.shape
        # (n_images, n_classes, height, width) * (n_images, [n_classes, 3], height, width) =>
        # => (n_images, 3, height, width)
        colorize_matrix = torch.stack(class_colors[:n_classes], dim=0)
        _y = y[:n_images].permute(0, 2, 3, 1).unsqueeze(3)
        colorize_matrix = colorize_matrix.reshape(1, 1, 1, n_classes, 3).expand(n_images, height, width, n_classes, 3)
        y_colored = (_y @ colorize_matrix).squeeze(3).permute(0, 3, 1, 2)

        resize_transform = Resize((y.shape[2] * Settings.output_downscale, y.shape[3] * Settings.output_downscale),
                                  interpolation=torchvision.transforms.InterpolationMode('nearest'))
        grid = torchvision.utils.make_grid(resize_transform(y_colored), nrow=n_cols)
        grid_numpy = grid.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        im = Image.fromarray(grid_numpy)
        im.save(f'images/{batch_idx}_output.png')

    if y_pred is not None:
        y_pred = y_pred.to('cpu')
        bs, n_classes, height, width = y_pred.shape
        # y_pred = F.one_hot(y_pred.softmax(dim=1).argmax(dim=1), num_classes=n_classes)
        colorize_matrix = torch.stack(class_colors[:n_classes], dim=0)
        _y_pred = y_pred[:n_images].permute(0, 2, 3, 1).unsqueeze(3)
        colorize_matrix = colorize_matrix.reshape(1, 1, 1, n_classes, 3).expand(n_images, height, width, n_classes, 3)
        y_pred_colored = (_y_pred @ colorize_matrix).squeeze(3).permute(0, 3, 1, 2)

        resize_transform = Resize((y_pred.shape[2] * Settings.output_downscale,
                                   y_pred.shape[3] * Settings.output_downscale),
                                  interpolation=torchvision.transforms.InterpolationMode('nearest'))
        grid = torchvision.utils.make_grid(resize_transform(y_pred_colored), nrow=n_cols)
        grid_numpy = grid.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        im = Image.fromarray(grid_numpy)
        im.save(f'images/{batch_idx}_prediction.png')

    ''' logit_threshold = - np.log(1 / np.clip(0.5, Settings.eps, 1 - Settings.eps) - 1)
        y_pred_bin = y_pred_bin.detach().to('cpu')
        pred_mask = cv2.resize(np.array(torch.where(y_pred_bin[0] > logit_threshold, 1, 0) * 255, dtype=np.uint8),
                               image.shape[:2], interpolation=cv2.INTER_NEAREST)
        wpred_mask = cv2.resize(extract_contours_and_boxes(y_pred_bin[0, 0]), image.shape[:2], interpolation=cv2.INTER_NEAREST)
        # image[..., 0][pred_mask != 0] = 255

        result = Image.fromarray(image)
        mask = Image.fromarray(mask)
        pred_mask = Image.fromarray(pred_mask)
        wpred_mask = Image.fromarray(wpred_mask)

        if not os.path.exists('images'):
            os.mkdir('images')

        result.save('images/' + str(batch_idx) + 'image' + '.png')
        mask.save('images/' + str(batch_idx) + 'mask' + '.png')
        pred_mask.save('images/' + str(batch_idx) + 'pred' + '.png')
        wpred_mask.save('images/' + str(batch_idx) + 'wpred' + '.png')'''


def postprocess(prediction: torch.Tensor, means: torch.Tensor, n_instances: int, bandwidth: float, norm: int):
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

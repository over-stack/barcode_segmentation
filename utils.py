import torch
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
                      position=A.PadIfNeeded.PositionType.TOP_LEFT, border_mode=cv2.BORDER_CONSTANT, value=(155, 155, 155)),
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
                      position=A.PadIfNeeded.PositionType.TOP_LEFT, border_mode=cv2.BORDER_CONSTANT, value=(155, 155, 155)),
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


def visualize(x, y, y_pred: torch.Tensor, batch_idx):
    x = x.detach().to('cpu')
    image = np.array(torch.permute(x[0], dims=(1, 2, 0)), dtype=np.float32)
    image = 255 * (image * np.array(Settings.std).reshape(1, 1, 3) + np.array(Settings.mean).reshape(1, 1, 3))
    image = image.astype(np.uint8)

    y = y.detach().to('cpu')
    mask = cv2.resize(np.array(y[0] * 255, dtype=np.uint8), image.shape[:2], interpolation=cv2.INTER_NEAREST)
    # image[..., 1][mask != 0] = 255
    # pred_mask = cv2.resize(extract_contours_and_boxes(y_pred[0, 0]), image.shape[:2], interpolation=cv2.INTER_NEAREST)

    logit_threshold = - np.log(1 / np.clip(0.5, Settings.eps, 1 - Settings.eps) - 1)
    y_pred = y_pred.detach().to('cpu')
    pred_mask = cv2.resize(np.array(torch.where(y_pred[0, 0] > logit_threshold, 1, 0) * 255, dtype=np.uint8), image.shape[:2], interpolation=cv2.INTER_NEAREST)
    wpred_mask = cv2.resize(extract_contours_and_boxes(y_pred[0, 0]), image.shape[:2], interpolation=cv2.INTER_NEAREST)
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
    wpred_mask.save('images/' + str(batch_idx) + 'wpred' + '.png')

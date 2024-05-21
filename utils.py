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
import xml.etree.ElementTree as ET
from torchvision.transforms import Resize
import cv2
from tqdm import tqdm

from collections import defaultdict
from config import Settings


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
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = Settings.seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_transform():
    transform = A.Compose([
        A.LongestMaxSize(max_size=Settings.width),
        A.PadIfNeeded(min_height=Settings.height, min_width=Settings.width,
                      position=A.PadIfNeeded.PositionType.TOP_LEFT),
        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=30, p=0.85),
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


def postporcess_base(prediction: torch.Tensor, prob_threshold: float = 0.5):
    """
    :param prediction: (bs, 1 + n_classes, height, width)
    :param prob_threshold: float[0, 1]
    :return: (bs, 1 + n_classes, height, width)
    """

    # prediction channels:
    # 0 - objects mask (inverse background)
    # 1 ... n - classes masks

    bin_logits = prediction[:, 0]
    logit_threshold = - np.log(1 / np.clip(prob_threshold, Settings.eps, 1 - Settings.eps) - 1)
    bin_logits = torch.where(bin_logits > logit_threshold, 1, 0)

    classes = prediction[:, 1:].argmax(dim=1) + 1
    result = (classes * bin_logits).to(torch.int64)
    result = F.one_hot(result, num_classes=prediction.shape[1]).permute(0, 3, 1, 2)

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
    # bs, n_loc, n_dims
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(bs, n_loc, n_dims)
    # bs, n_loc, n_instances, n_dims
    prediction = prediction.unsqueeze(2).expand(bs, n_loc, n_instances, n_dims)
    # n_loc, n_instances, n_dims
    means = means.unsqueeze(0).expand(n_loc, n_instances, n_dims)
    # bs, n_loc, n_instances, n_dims
    means = means.unsqueeze(0).expand(bs, n_loc, n_instances, n_dims)

    eps = 1e-5
    # bs, n_loc
    result = torch.clamp(torch.norm((prediction - means), norm, 3) - bandwidth, min=0.0).argmin(dim=2)
    result = result.reshape(bs, height, width)
    result = F.one_hot(result, num_classes=n_instances).permute(0, 3, 1, 2)

    # bs, n_instances, height, width
    return result.to(torch.int64)


def postprocess_objects(pred_post: torch.Tensor, min_object_area: int = 20):
    """
    :param pred_post: (bs, n_instances, height, witdth)
    :param min_object_area:
    :return:
    """

    pred_post = pred_post.detach().to('cpu')
    objects_masks = (pred_post[:, 0] != 1) * 255  # * 255 ???
    classes = pred_post[:, 1:].argmax(dim=1).numpy()

    result_masks = list()
    result_boxes = list()
    result_labels = list()
    for image_idx in range(pred_post.shape[0]):
        contours, _ = cv2.findContours(np.array(objects_masks[image_idx], dtype=np.uint8), mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda contour: cv2.contourArea(contour) > min_object_area, contours))
        rectangles = [cv2.minAreaRect(contour) for contour in contours]
        boxes = [np.int0(cv2.boxPoints(rect)) for rect in rectangles]
        labels = list()

        empty_mask = np.zeros(objects_masks[image_idx].shape, dtype=np.uint8)
        image_result = np.zeros(objects_masks[image_idx].shape, dtype=np.uint8)
        for i in range(len(boxes)):
            object_mask = cv2.drawContours(empty_mask, boxes, i, color=1, thickness=-1)
            object_class = np.bincount(classes[image_idx][object_mask != 0]).argmax()
            image_result[object_mask != 0] = object_class + 1
            labels.append(Settings.classes_reverse[object_class + 1])

        result_masks.append(image_result)
        result_boxes.append(np.array(boxes))
        result_labels.append(labels)

    result_masks = torch.from_numpy(np.stack(result_masks, axis=0)).to(torch.int64).to(pred_post.device)  # cpu
    result_masks = F.one_hot(result_masks, num_classes=pred_post.shape[1]).permute(0, 3, 1, 2)

    return result_masks, result_boxes, result_labels


def parse_xml_for_barcodes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    barcodes = {
        'type_ids': list(),
        'bboxes': list()
    }

    for barcode in root.findall(".//Barcode"):
        type_ = barcode.get('Type')
        bbox = list()

        points = barcode.findall('.//Point')
        for point in points:
            bbox.append([int(round(float(point.get('X')))), int(round(float(point.get('Y'))))])

        barcodes['type_ids'].append(Settings.classes[type_])
        barcodes['bboxes'].append(bbox)

    return barcodes


def box_to_mask(image, box: np.ndarray, color):
    result = cv2.drawContours(image, [box], 0, color, -1)
    return result


def calc_stats_images(train_img_dir):
    filenames = os.listdir(train_img_dir)
    images_path = [os.path.join(train_img_dir, filename) for filename in filenames]
    images_rgb = list()

    for img in tqdm(images_path):
        images_rgb.append(np.array(Image.open(img).convert('RGB').getdata()) / 255.)

    means = []
    for image_rgb in tqdm(images_rgb):
        means.append(np.mean(image_rgb, axis=0))
    mean = np.mean(means, axis=0)

    variances = []
    for image_rgb in tqdm(images_rgb):
        var = np.mean((image_rgb - mean) ** 2, axis=0)
        variances.append(var)
    std = np.sqrt(np.mean(variances, axis=0))

    return mean, std


# need fix (optional)
def calc_weights(data_dir):
    types = defaultdict(int)
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        for barcode in parse_xml_for_barcodes(file_path):
            types[barcode['type']] += 1

    weights = list()
    n_samples = sum(types.values())
    for key, value in types.items():
        weights.append(n_samples / (len(types) * value))

    return torch.tensor(weights)

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from utils import Mode
from config import Settings
import torchvision

from utils import parse_xml_for_barcodes, box_to_mask
from config import Settings


class BarcodeDataset(Dataset):
    def __init__(self, mode, images_path, masks_path, filenames, transform=None):
        self.mode = mode
        self.images_path = images_path
        self.masks_path = masks_path
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        image_path = os.path.join(self.images_path, filename)
        image = np.array(Image.open(image_path).convert('RGB'))
        if self.mode != Mode.TEST:
            mask_path = os.path.join(self.masks_path, filename[:-3] + 'png')
            mask = np.array(Image.open(mask_path).convert('L')) // 255

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            resize_transform = torchvision.transforms.Resize((mask.shape[0] // Settings.output_downscale,
                                                              mask.shape[1] // Settings.output_downscale),
                                                             interpolation=torchvision.transforms.InterpolationMode(
                                                                 'nearest'))

            mask = resize_transform(mask.unsqueeze(0)).to(dtype=torch.int64)
            mask = F.one_hot(mask).permute(0, 3, 1, 2).squeeze(0)
            # (for future) CrossEntropyLoss is faster with indices (not one-hot)

            n_objects = torch.tensor(mask.shape[0])

            return image, mask, n_objects
        else:
            if self.transform:
                image = self.transform(image=image)['image']

            return image


class BarcodeDatasetXML(Dataset):
    def __init__(self, mode, images_path, masks_path, filenames, transform=None):
        self.mode = mode
        self.images_path = images_path
        self.labels_path = masks_path
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        image_path = os.path.join(self.images_path, filename)
        image = np.array(Image.open(image_path).convert('RGB'))
        if self.mode != Mode.TEST:
            label_path = os.path.join(self.labels_path, filename.rsplit('.', 1)[0] + '.xml')
            barcodes = parse_xml_for_barcodes(label_path)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            type_ids = barcodes['type_ids']
            bboxes = barcodes['bboxes']

            for i in range(len(type_ids)):
                # type_ids[i] => 1
                mask = box_to_mask(mask, np.array(bboxes[i]), 1)

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            resize_transform = torchvision.transforms.Resize((mask.shape[0] // Settings.output_downscale,
                                                              mask.shape[1] // Settings.output_downscale),
                                                             interpolation=torchvision.transforms.InterpolationMode(
                                                                 'nearest'))

            mask = resize_transform(mask.unsqueeze(0)).to(dtype=torch.int64)
            mask = F.one_hot(mask, num_classes=Settings.num_classes + 1).permute(0, 3, 1, 2).squeeze(0)
            # (for future) CrossEntropyLoss is faster with indices (not one-hot)

            n_objects = torch.tensor(mask.shape[0])

            return image, mask, n_objects
        else:
            if self.transform:
                image = self.transform(image=image)['image']

            return image

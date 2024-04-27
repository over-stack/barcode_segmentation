from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from utils import Mode
from config import Settings
import torchvision


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
            mask_path = os.path.join(self.masks_path, filename)
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

            n_objects = torch.tensor(mask.shape[0])

            return image, mask, n_objects
        else:
            if self.transform:
                image = self.transform(image=image)['image']

            return image

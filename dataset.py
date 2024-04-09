from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from utils import Mode
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
            mask_path = os.path.join(self.masks_path, filename)
            mask = np.array(Image.open(mask_path).convert('L')) // 255

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            return image, mask[::4, ::4].to(dtype=torch.int8)
        else:
            if self.transform:
                image = self.transform(image=image)['image']

            return image

import torch
import random
import numpy as np
import os

from sklearn.model_selection import train_test_split

from model import Model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from train import train_model
from utils import Settings, Mode, get_train_transform
from dataset import BarcodeDataset
import matplotlib.pyplot as plt
from albumentations.augmentations import functional as AF


def main():
    model = Model(3, 1, 24)
    train_model(model)

    '''filenames = sorted(os.listdir(Settings.dataset_path + '/IMAGES'))
    train_filenames, val_filenames = train_test_split(filenames, train_size=0.8, shuffle=True,
                                                      random_state=Settings.seed)
    train_transform = get_train_transform()
    train_dataset = BarcodeDataset(Mode.TRAIN, Settings.dataset_path + '/IMAGES', Settings.dataset_path + '/MASKS',
                                   train_filenames, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=Settings.batch_size,
                                  shuffle=True, num_workers=Settings.num_workers,
                                  pin_memory=Settings.pin_memory, persistent_workers=True)

    fig, axs = plt.subplots(2, 2)
    for X, y in train_dataloader:
        img1 = np.array(torch.permute(X[0], dims=[1, 2, 0]), dtype=np.float32)
        img2 = AF.normalize(
            img=img1,
            mean=[-m / s for m, s in zip(Settings.mean, Settings.std)],
            std=[1 / s for s in Settings.std]
        )
        img2 = 255 * (img1 * np.array(Settings.std).reshape(1, 1, 3) + np.array(Settings.mean).reshape(1, 1, 3))
        img2 = img2.astype(np.uint8)
        print(img1.min(), img1.max())
        print(img2.min(), img2.max())
        mask1 = y[0]
        mask2 = y[1]
        axs[0, 0].imshow(img1)
        axs[0, 1].imshow(mask1)
        axs[1, 0].imshow(img2)
        axs[1, 1].imshow(mask2)
        break
    plt.show()'''


if __name__ == "__main__":
    main()

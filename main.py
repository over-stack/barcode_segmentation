import torch
import random
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    # num_classes without background
    model = Model(Settings.in_channels, Settings.num_classes, Settings.embedding_dims, Settings.num_filters)
    train_model(model, Settings.base_loss_weight, Settings.embedding_loss_weight)

    # prediction channels:
    # 0 - objects mask (inverse background)
    # 1 ... n - classes masks

    # target channels:
    # 0 ... n - classes - masks (0 - background mask)


if __name__ == "__main__":
    main()

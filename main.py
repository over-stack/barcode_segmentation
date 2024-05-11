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
    model = Model(3, 1, 2, 24)
    train_model(model, base_loss_weight=1, embedding_loss_weight=1)


if __name__ == "__main__":
    main()

import os
import sys
import ast
import datetime
import gin
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from argparse import ArgumentParser
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

np.set_printoptions(suppress=True)
plt.rcParams.update({'font.size': 14})
ImageFile.LOAD_TRUNCATED_IMAGES = True

EPSILON = 1e-12
Y_MISSING_VALUE = -1

class DataFrameDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            x = Image.open(self.df.index[idx])
            if self.transform is not None:
                x = self.transform(x)
            y = self.df.iloc[idx].values
            return x, y
        except: # This is not good practice
            return None

class RunningAverage:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, values):
        self.sum += values.sum().item()
        self.count += len(values)

    def __call__(self):
        if self.count == 0:
            return np.nan
        else:
            return self.sum / self.count

def set_seed(seed):
    '''
    Ensure reproducibility.
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_file(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

def load_file(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)

def remove_file(fpath):
    try:
        os.remove(fpath)
    except OSError:
        pass

def write(fpath, text):
    with open(fpath, 'a+') as f:
        f.write(text + '\n')

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def total_variation(p, q):
    return np.linalg.norm(p - q, ord=1) / 2

def filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
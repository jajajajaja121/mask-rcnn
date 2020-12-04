import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from dataset.dataset import MaskDataset
from model.resnet import res18

def train():
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

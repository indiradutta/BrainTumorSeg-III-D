import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchio as tio

import os
import h5py
import cv2
import pandas as pd
import numpy as np
import scipy
import pickle
import re
from tqdm import tqdm

import nibabel as nib

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import warnings
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

def train(model, trainloader, lr, epochs, optimizer):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()

    losses = []
    dice_scores = []

    model.train()

    for epoch in tqdm(range(epochs)):

        for idx, (x,y) in enumerate(trainloader):
            
            for i in range(len(x)):            
              inputs, targets = x[i].float(),y[i].float()
              inputs, targets = inputs.cuda(),targets.cuda()
              targets =  targets.permute(0,4,1,2,3)
              optimizer.zero_grad()

              output_main = model(inputs)

              d_loss, d_score = dice_loss(output_main, targets.detach(), multiclass=True)
              d_loss.backward()
              optimizer.step()

        losses.append(d_loss.item())
        dice_scores.append(d_score)

        print('Epoch: ',str(epoch),'Dice Loss: ',str(d_loss.item()), 'Dice Score: ', str(d_score))

    model_data = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'loss': losses,'dice': dice_scores}

    torch.save(model_data, 'pspnet3D.pth')  
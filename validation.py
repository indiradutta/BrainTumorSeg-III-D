import torch
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import pickle
import argparse
import os

from data.preprocessing import Preprocessing

from src.models.ARUNET.arunet3D import ARUNET, Block, Attention
from src.models.LinkNet.linknet3D import LinkNet
from src.models.PSPNet.pspnet3D import PSPNet

from src.train.val import validation

with open('imgs.pkl','rb') as file:
    d = pickle.load(file)


__models__ = ["arunet3d",
              "linknet3d",
              "pspnet3d"
             ]
  
parser = argparse.ArgumentParser(description='3D Semantic Segmentation')
parser.add_argument("--model_name", default="pspnet3d", type=str, required=True, help="name of the model")
parser.add_argument("--dataset_path", type=str, required=True, help="path to training data")
parser.add_argument("--epochs", default=200, type=int, required=True, help="number of iterations")
args = parser.parse_args()

# print(args.model_name)

if args.model_name not in __models__:
    raise ValueError("{} not supported yet. {} are the supported models".format(model, __models__))
    
elif args.model_name == "arunet3d":
    model = ARUNET(Block, Attention, 1, [64,128,256,512])
    epochs = args.epochs
    val_d = Preprocessing(im_path=args.dataset_path, l1=d[251:], im_dim = (64,64), test=False)
    vdata = DataLoader(val_d, batch_size=4, shuffle=True)
 
elif args.model_name == "linknet3d":
    model = LinkNet()
    epochs = args.epochs
    val_d = Preprocessing(im_path=args.dataset_path, l1=d[251:], im_dim = (224,224), test=False)
    vdata = DataLoader(val_d, batch_size=4, shuffle=True)
    
elif args.model_name == "pspnet3d":
    model = PSPNet()
    epochs = args.epochs
    val_d = Preprocessing(im_path=args.dataset_path, l1=d[251:], im_dim = (128,128), test=False)
    vdata = DataLoader(val_d, batch_size=2, shuffle=True)
    
# print(model)
print("Validation...")
validation(model, vdata, epochs)

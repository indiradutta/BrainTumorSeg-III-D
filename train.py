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

from src.train.aru_train import train_arunet
from src.train.link_train import train_linknet
from src.train.psp_train import train_pspnet


with open('imgs.pkl','rb') as f:
    d = pickle.load(f)


__models__ = ["arunet3d",
              "linknet3d",
              "pspnet3d"
             ]

parser = argparse.ArgumentParser(description='3D Semantic Segmentation')
parser.add_argument("--model_name", default="pspnet3d", type=str, required=True, help="name of the model")
parser.add_argument("--dataset_path", type=str, required=True, help="path to training data")
parser.add_argument("--epochs", default=200, type=int, required=True, help="number of iterations")
args = parser.parse_args()

print(args.model_name)
if args.model_name not in __models__:
    raise ValueError("{} not supported yet. {} are the supported models".format(model, __models__))

elif args.model_name == "arunet3d":
    ARU = ARUNET(Block, Attention, 1, [64,128,256,512])
    epochs = args.epochs
    train_d = Preprocessing(im_path=args.dataset_path, l1=d[:250], im_dim = (64,64), test=False)
    tdata = DataLoader(train_d, batch_size=4, shuffle=True)
    opt = torch.optim.RMSprop(ARU.parameters(),lr=0.0001)
    
    train_arunet(ARU, tdata, opt, epochs)

elif args.model_name == "linknet3d":
    LINK = LinkNet()
    epochs = args.epochs
    train_d = Preprocessing(im_path=args.dataset_path, l1=d[:250], im_dim = (224,224), test=False)
    tdata = DataLoader(train_d, batch_size=4, shuffle=True)
    opt = torch.optim.RMSprop(LINK.parameters(),lr=0.00001)
    
    train_linknet(LINK, tdata, opt, epochs)
    
elif args.model_name == "pspnet3d":
    PSP = PSPNet()
    epochs = args.epochs
    train_d = Preprocessing(im_path=args.dataset_path, l1=d[:250], im_dim = (128,128), test=False)
    tdata = DataLoader(train_d, batch_size=2, shuffle=True)
    opt = torch.optim.RMSprop(PSP.parameters(),lr=0.0001)
    
    train_pspnet(PSP, tdata, opt, epochs)

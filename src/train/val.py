import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import tqdm

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

from src.models.ARUNET.arunet3D import ARUNET
from src.models.LinkNet.linknet3D import LinkNet
from src.models.PSPNet.pspnet3D import PSPNet

from .dice import DiceLoss
from .plotting import plot

def validation(model, loader, epochs):

    losses = []
    dice_scores = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
    
    model.eval()

    for idx in range(epochs):
        
        with torch.no_grad():
            
            for _,(x,y) in enumerate(loader):
                
                for i in range(len(x)): 
                    
                    inputs, targets = x[i].float(), y[i].float()
                    inputs, targets = inputs.cuda(), targets.cuda()
                    targets =  targets.permute(0,4,1,2,3)

                    out = model(inputs)                    
                    dice = DiceLoss()
                    loss, score = dice.dice_loss(out, targets.detach(), multiclass=True)
            
        losses.append(loss.item())
        dice_scores.append(score.item())

        print('Epoch: ',str(e),'Dice Loss: ',str(loss.item()),'Dice Score: ',str(score.item()))

    plot(losses,dice_scores)

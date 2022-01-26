import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

from src.models.PSPNet.pspnet3D import PSPNet
from .dice import DiceLoss
from .plotting import plot

def train_pspnet(model, trainloader, optimizer, epochs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()

    losses = []
    dice_scores = []

    model.train()

    for epoch in range(epochs):

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
    
    plot(losses, dice_scores)

    torch.save(model_data, 'pspnet3D.pth')  
    print("done :)")

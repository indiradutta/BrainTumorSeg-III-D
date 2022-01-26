import tqdm
from tqdm import trange
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

from src.models.LinkNet.linknet3D import LinkNet
from .dice import DiceLoss
from .plotting import plot

def train_linknet(net, loader, opt, epochs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        net.cuda()      # shifting model to cuda for training
    
    #net = net.cuda()   
    
    losses = []  # storing the dice loss
    dsc = []   # storing the dice scores/coefficient

    #scheduler = StepLR(opt,10,0.1)
    
    net.train()

    for epoch in range(epochs):
        
        for _,(x,y) in enumerate(loader):
            
            #print(len(x))
            
            #for each image - 3 images per training sample
            for i in range(len(x)): 
                
                inputs, targets = x[i].float(), y[i].float()
                inputs, targets = inputs.cuda(), targets.cuda()
                    
                opt.zero_grad()
                
                out = net(inputs)
                
                dice = DiceLoss()
                loss, score = dice.dice_loss(out, targets.detach(), multiclass=True)
                loss.backward()

                opt.step()  
              
        losses.append(loss.item())
        dsc.append(score.item())
       
        print('Epoch: ',str(epoch),'Dice Loss: ',str(loss.item()),'Dice Score: ',str(score.item())) 

    model_data = {'model': net.state_dict(),'optimizer': opt.state_dict(),'loss': losses,'dice': dsc} 

    plot(losses, dsc)

    torch.save(model_data,'linknet3d.pth')
    print("done :)")

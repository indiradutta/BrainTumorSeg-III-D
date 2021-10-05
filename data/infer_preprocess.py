import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import nibabel as nib
import cv2

class Infer_Preprocess(Dataset):
  
    """
    This preprocessing class is used solely for mod preprocessing the images during inference
    
    """

    def __init__(self):

        super(Infer_Preprocess, self).__init__()

    def mod_preprocess(self, img, img_dim = (128, 128)):
      
        """
        returns the normalized image with img_dim dimension and 64 slices
        
        Parameter:
        
        - img: array of the input image
        
        - img_dim(default: (128, 128)): dimension of the image (set for pspnet3d by default)
        
        """
      
        img = np.asarray(img)
        img1 = cv2.resize(img, img_dim)
        xpre = np.zeros((1, img_dim[0], img_dim[1],64))
        c = 0

        for i in range(40,104):

            slic = img1[:,:,i]
            
            #mean 0
            centr = slic - np.mean(slic)
            if np.std(centr)!=0:
                centr = centr/np.std(centr)

            xpre[0,:,:,c] = centr
            c += 1

        return xpre

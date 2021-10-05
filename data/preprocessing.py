from tqdm import tqdm
import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import nibabel as nib
import cv2

class Preprocessing(Dataset):

    def __init__(self, im_path, l1, im_dim = (128,128), bs=1, test=False):

        """
        Parameters:

        - im_path: the root to the directory containing the images

        - l1: list of the names of each training sample
        
        - bs(default: 1): batch_size (no. of images passed at once)
        
        - im_dim(default: (128, 128)): dimension of the passed images, should be changed with respect to architectures ((64, 64) for arunet, (224, 224) for linknet) 
        
        """

        super(Preprocessing,self).__init__()

        self.root = im_path
        self.dim = im_dim
        self.files = l1
        self.bs = bs
        self.count = 0
        self.test = test

    def __len__(self):
      
        """
        returns the length of the list l1
        
        """
      
        return len(self.files)

    def mod_preprocess(self, img):
      
        """
        returns the normalized image with 128 slices
        
        Parameter:
        
        - img: array of the input image
        
        """
      
        img = np.asarray(img)
        img1 = cv2.resize(img, self.dim)
        xpre = np.zeros((1, self.dim[0], self.dim[1], 64))
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


    def mask_preprocess(self, msk):
      
        """
        returns the binary version of the mask - tumor(1) or no tumor(0)
        
        Parameters:
        
        - msk: passed as the 'mod_preprocess'-ed image
        
        """
      
        img1 = np.asarray(msk)
        img1 = cv2.resize(img1, self.dim)
        img1 = img1[:,:,40:104]
        
        img1[img1==4] = 3
        img = F.one_hot(torch.from_numpy(img1).to(torch.int64), num_classes=4)
        #print(img.size())
        return img.permute(3,0,1,2)


    def flip(self, img):
      
        """
        returns a randomly flipped version of the 'mod_preprocess'-ed image
        
        Parameters:
        
        - img: passed as the 'mod_preprocess'-ed image
        
        """
      
        transform = tio.RandomFlip(axes=('LR',))
        transformed_img = transform(img)
        return transformed_img


    def affine_transformation(self, img):
      
        """
        returns the affine transformation of the 'flip'-ed image
        
        Parameters:
        
        - img: passed as the randomly 'flip'-ed image
        
        """
      
        transform = tio.RandomAffine(
          scales = (0.9, 1.2),
          degrees = 30,
          isotropic = True,
          image_interpolation = 'linear',
          )
        
        transformed_img = transform(img)
        return transformed_img


    def __getitem__(self, idx):
      
        """
        returns the completely preprocessed input and output images for the model
        
        Parameters:
        
        - idx: list index value
        
        """
      
        im_file = self.files[idx]
        im_path = os.path.join(self.root, im_file)
        img_mod = os.listdir(im_path)
        x, y = [], []
       
        for i in img_mod:
            modality = i.strip().split('.')[0].split('_')[3]

            if modality == 'seg':
              
                img = nib.load(im_path+'/'+im_file+'_'+modality+'.nii').get_fdata()
                img = np.asarray(img)
                img = self.mask_preprocess(img)
                y.append(img)

            elif modality == 'flair':
              
                img = nib.load(im_path+'/'+im_file+'_'+modality+'.nii').get_fdata()
                img = np.asarray(img)
                img = self.mod_preprocess(img)
                
                if self.count%20 == 0:
                    img = self.flip(img)
                    img = self.affine_transformation(img)
                    x.append(img)
                    
                else: 
                    x.append(img)
                    
                self.count += 1

        return x,y

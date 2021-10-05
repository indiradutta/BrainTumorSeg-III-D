import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import numpy as np
import os
import json
import gdown
import time
import cv2
import nibabel as nib

from .data.infer_preprocess import Infer_Preprocess

from .src.models.ARUNET.arunet3D import ARUNET, Block, Attention
from .src.models.LinkNet.linknet3D import LinkNet
from .src.models.PSPNet.pspnet3D import PSPNet


__PREFIX__ = os.path.dirname(os.path.realpath(__file__))


__models__ = ["arunet3d",
              "linknet3d",
              "pspnet3d"
             ]

def available_models():

    """ Returns list of all supported models """

    return __models__

class Seg(object):

    def __init__(self, img_path, model = "pspnet3d", img_dim = (128,128)):

        if model not in __models__:
            raise ValueError("{} not supported yet. {} are the supported models".format(model, available_models()))

        self.img_path = img_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.arch = model
        self.dim = img_dim
    
    def inference(self, set_weight_dir = "pspnet3d.pth", path = 'result.png'):

        set_weight_dir = __PREFIX__ + "/weights/" + set_weight_dir


        ''' saving generated images in a directory '''

        def save_image(path):

            if os.path.exists(path):
                print("Found directory for saving generated images")
                return 1

            else:
                print("Directory for saving images not found, making a directory named 'result_img'")
                os.mkdir(path)
                return 1

        ''' checking if weights are present '''

        def check_weights(set_weight_dir):

            if os.path.exists(set_weight_dir):
                print("Found weights")
                return 1

            else:
                print("Downloading weights")
                download_weights()


        ''' downloading weights if not present '''

        def download_weights():

            with open(__PREFIX__+"/config/"+self.arch+"_weight.json") as fp:

                json_file = json.load(fp)

                if not os.path.exists(__PREFIX__+"/weights/"):
                    os.mkdir(__PREFIX__+"/weights/")

                url = 'https://drive.google.com/uc?id={}'.format(json_file[self.arch+'.pth'])
                gdown.download(url, __PREFIX__+"/weights/"+self.arch+".pth", quiet=False)
                #set_weight_dir = "linknet3d.pth"

                print("Download finished")

        check_weights(set_weight_dir)
        
        ''' displaying the result and saving the image '''

        def show(testim, res):

            plt.figure(figsize=(15,4))
            
            plt.subplot(1,5,1)
            plt.imshow(testim[:,:,77])
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1,5,2)
            plt.imshow(res[:,:,45,0])
            plt.title('No tumor')
            plt.axis('off')
            
            plt.subplot(1,5,3)
            plt.imshow(res[:,:,45,1])
            plt.title('Class 1')
            plt.axis('off')
            
            plt.subplot(1,5,4)
            plt.imshow(res[:,:,45,2])
            plt.title('Class 2')
            plt.axis('off')
            
            plt.subplot(1,5,5)
            plt.imshow(res[:,:,46,3])
            plt.title('Class 3')
            plt.axis('off')
            
            #plt.show()
            plt.savefig(path)

        testim = nib.load(self.img_path).get_fdata()
        
        if self.arch == "arunet3d":
            model = ARUNET(Block, Attention, 1, [64, 128, 256, 512])
        elif self.arch == "linknet3d":
            model = LinkNet()
        elif self.arch == "pspnet3d":
            model = PSPNet()
        else:
            if self.arch not in __models__:
                raise ValueError("{} not supported yet. {} are the supported models".format(self.arch, available_models()))

        img = Infer_Preprocess.mod_preprocess(1, testim, self.dim)
       
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)
        img = img.to(self.device)

        model.load_state_dict(torch.load(set_weight_dir, map_location=self.device)['model'])
        model = model.to(self.device)
        res = model(img.float())

        res = res[0].permute(1,2,3,0).cpu().detach()
        
        show(testim, res)

""" 
This test file may be run from the terminal to get the inferences for a single .nii mri image. The image used here is provided in the repo itself, however you may 
use any .nii image of your choice.

The model for segmentation has been set to pspnet3d by default. You may use any model with any dimension of your choice, as long as your computation is compatible 
with it.

"""

from segment import Seg

obj = Seg('results/test.nii', model = "pspnet3d", img_dim = (128,128))
obj.inference(set_weight_dir = "pspnet3d.pth", path = 'result.png')

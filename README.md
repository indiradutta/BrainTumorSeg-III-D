# Utilizing Attention, Linked Blocks, and Pyramid Pooling to Propel Brain Tumor Segmentation in 3D
Brain tumor segmentation involves identifying the tumorous region in the brain (pixels indicating the tumorous cells in the MRI scans). Hence, for the purpose of segmenting brain tumors, we present three novel 3D architectures.

![epi](https://user-images.githubusercontent.com/66861243/136070815-ea8bf6cd-517a-409a-b3a4-f01ad0c93b2c.jpg)


## Abstract
We present an approach to detect and segment tumorous regions of the brain by establishing three varied segmentation architectures for multiclass semantic segmentation along with data specific customizations like residual blocks, soft attention mechanism, pyramid pooling, linked architecture and 3D compatibility to work with 3D brain MRI images. The proposed segmentation architectures namely, Attention Residual UNET 3D also referred to as AR-UNET 3D, LinkNet 3D and PSPNet 3D, segment the MRI images and succeed in isolating three classes of tumors. By assigning pixel probabilities, each of these models differentiates between pixels belonging to tumorous and non-tumorous regions of the brain. By experimenting and observing the performance of each of the three architectures using metrics like Dice loss and Dice score, on the BraTS2020 dataset, we successfully establish quality results.

## Research Overview
1. AR-UNET 3D - The Attention Residual UNET 3D or AR-UNET 3D is a modification upon the existing Residual U-Net and Attention U-Net, both of which operate in 2D. AR-UNET 3D makes use of the residual blocks from ResNet which help in maintaining skip connections using identity mappings while the proposed soft attention mechanism provides an added advantage by weighing the more important features, heavily.

2. LinkNet 3D - The LinkNet 3D is a modification upon the existing LinkNet in 2D. LinkNet 3D makes use of residual blocks from ResNet18 3D in its encoder for feature extraction, and links the output from each encoder block to its corresponding decoder block to account for lost spatial information due to multiple downsampling. 

3. PSPNet 3D - The PSPNet 3D is a modification upon the existing PSPNet in 2D. PSPNet 3D makes use of a 3D Pyramid Pooling Module for interpolating the 3D feature maps into different resolutions that facilitates efficient extraction of spatial information and global context capturing. This helps the model learn about the spatial context associated with different classes of the image.

## Installation and Quick Start
To use the repo and run inferences, please follow the guidelines below

- Cloning the Repository: 

        $ git clone https://github.com/indiradutta/Utilizing-Attention-Linked-Blocks-And-Pyramid-Pooling-To-Propel-Brain-Tumor-Segmentation-In-3D
        
- Entering the directory: 

        $ cd Utilizing-Attention-Linked-Blocks-And-Pyramid-Pooling-To-Propel-Brain-Tumor-Segmentation-In-3D/
        
- Setting up the Python Environment with dependencies:

        $ pip install -r requirements.txt

- Running the file for inference:

        $ python test.py
 
Running the `test.py` file, downloads the weights and segments a single `.nii` image using the `pspnet3d` model by default. If you want to use any other model please consider making changes to the image dimensions and the model while initializing the `Segment` class. In order to use `linknet3d` or `arunet3d` consider running the code snippets given below.

- For LinkNet 3D:
```python
from segment import Seg

# initializing the Seg object 
seg = Seg('results/test.nii', model = "linknet3d", img_dim = (224,224))
seg.inference(set_weight_dir = "linknet3d.pth", path = 'result.png')
```

- For AR-UNET 3D:
```python
from segment import Seg

# initializing the Seg object 
seg = Seg('results/test.nii', model = "arunet3d", img_dim = (64,64))
seg.inference(set_weight_dir = "arunet3d.pth", path = 'result.png')
```

## Results
- Training Loss and Score Progression 

Architecture | Loss and Score Progression
:-------------: | :---------: |
AR-UNET 3D | ![arunet_loss](https://user-images.githubusercontent.com/66861243/136073912-bb28ae11-3ff2-4771-8857-f29d01fc78a3.png) |
LinkNet 3D | ![linknet_loss](https://user-images.githubusercontent.com/66861243/136074028-2e4e77ad-de9e-4ee7-aab1-0f7d0784d0b4.png) |
PSPNet 3D | ![pspnet_loss](https://user-images.githubusercontent.com/66861243/136074196-c327e5c6-c8f3-4ecc-a4c6-deb1cae2964c.png) |


- Segmentation Results

Architecture | Result
:--------------------: | :-------------------: |
AR-UNET 3D | ![arunet_res](https://user-images.githubusercontent.com/66861243/136074804-2fcfbda4-9ae4-4f85-ad95-f3a6f02a2427.png) |
LinkNet 3D | ![linknet_res](https://user-images.githubusercontent.com/66861243/136074856-605644ba-9f3f-4a17-a4e3-efedd65c25c8.png) |
PSPNet 3D | ![pspnet_res](https://user-images.githubusercontent.com/66861243/136074937-0a6c4a0f-7320-471c-94c5-dca758851a75.png) |

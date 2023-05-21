<h1 align = "center">Attention Mechanism, Linked Networks, and Pyramid Pooling Enabled 3D Biomedical Image Segmentation.

## Abstract
We present an approach to detect and segment tumorous regions of the brain by establishing three varied segmentation architectures for multiclass semantic segmentation along with data specific customizations like residual blocks, soft attention mechanism, pyramid pooling, linked architecture and 3D compatibility to work with 3D brain MRI images. The proposed segmentation architectures namely, Attention Residual UNET 3D also referred to as AR-UNET 3D, LinkNet 3D and PSPNet 3D, segment the MRI images and succeed in isolating three classes of tumors. By assigning pixel probabilities, each of these models differentiates between pixels belonging to tumorous and non-tumorous regions of the brain. By experimenting and observing the performance of each of the three architectures using metrics like Dice loss and Dice score, on the BraTS2020 dataset, we successfully establish quality results.

## Research Overview
1. **AR-UNET 3D** - The Attention Residual UNET 3D or AR-UNET 3D is a modification upon the existing Residual U-Net and Attention U-Net, both of which operate in 2D. AR-UNET 3D makes use of the residual blocks from ResNet which help in maintaining skip connections using identity mappings while the proposed soft attention mechanism provides an added advantage by weighing the more important features, heavily.

2. **LinkNet 3D** - The LinkNet 3D is a modification upon the existing LinkNet whihc operated in 2D. LinkNet 3D makes use of residual blocks from ResNet18 3D in its encoder for feature extraction, and links the output from each encoder block to its corresponding decoder block to account for lost spatial information as a result of multiple downsampling. 

3. **PSPNet 3D** - The PSPNet 3D is a modification upon the existing PSPNet which operates in 2D. PSPNet 3D makes use of a 3D Pyramid Pooling Module for interpolating the 3D feature maps into different resolutions that facilitates efficient extraction of spatial information and global context capturing that helps it learn about the spatial context associated with different classes of the image.

All 3 segmentation models were trained using the [PyTorch](https://pytorch.org/docs/stable/index.html) with 250 images of batch size 4, for 200 iterations using the [RMSProp](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) optimization algorithm.

## Dataset
We have used the [Brain Tumor Segmentation 2020 (BraTS2020)](https://www.med.upenn.edu/cbica/brats2020/data.html) dataset for the brain tumor segmentation task in 3d space using our proposed architectures. All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe 
* native (T1) 
* post-contrast T1-weighted (T1Gd)
* T2-weighted (T2)
* T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes 

### Guidelines to download, setup and use the dataset
The BraTS2020 dataset maybe downloaded [here](https://www.kaggle.com/awsaf49/brats20-dataset-training-validation) as a file named *archive.zip*.

**Please write the following commands on your terminal to extract the file in the proper directory.**
```bash
  $ mkdir brats
  $ unzip </path/to/archive.zip> -d </path/to/brats>
```

## Installation and Quick Start
To use the repo and run inferences, please follow the guidelines below

- Cloning the Repository: 

        $ git clone https://github.com/indiradutta/BrainTumorSeg-III-D
        
- Entering the directory: 

        $ cd BrainTumorSeg-III-D/
        
- Setting up the Python Environment with dependencies:

        $ pip install virtualenv
        $ virtualenv venv
        $ source venv/bin/activate
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
The repo comes pre-installed with a training file `train.py`. If you want to re-train any of the models with more number of images or slices, please run the following command:
```bash
python train.py --model_name <model name> --dataset_path <path/to/MICCAI_BraTS2020_TrainingData/> --epochs <iterations>
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

## Citation
If this work is useful for your research, please consider citing through:
```
@INPROCEEDINGS{10051771,
  author={Ravi, Pooja and Roy, Srijarko and Dutta, Indira and Kottursamy, Kottilingam},
  booktitle={2022 IEEE/ACIS 23rd International Conference on Software Engineering, Artificial Intelligence, Networking and Parallel/Distributed Computing (SNPD)}, 
  title={Attention Mechanism, Linked Networks, and Pyramid Pooling Enabled 3D Biomedical Image Segmentation}, 
  year={2022},
  volume={},
  number={},
  pages={91-96},
  doi={10.1109/SNPD54884.2022.10051771}}
```

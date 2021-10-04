import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchio as tio

import cv2
import pandas as pd
import numpy as np

class Bottleneck(nn.Module):
    
    """
    Class for each Bottleneck in ResNet50
    
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        
        """
        Parameters:
        
        - in_channels: no. of input channels
        
        - out_channels: no. of output channels
        
        - kernel_size: kernel size for conv in each block
        
        - stride(default: 1): stride to be assigned to each block
        
        - dilation(default: 1): amount of dilation to be assigned to each block
        
        - downsample(default: None): downsampling module to be added
        
        - bias(default: False): boolean bias to be assigned to each block
        
        """
        
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * 4, kernel_size=1, stride=1, dilation=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet50_3D(nn.Module):
    
    """
    Main class for building the ResNet50
    
    """

    def __init__(self, block, layers=[3, 4, 6, 3], num_classes=4, base=True):
        super(ResNet50_3D, self).__init__()
        
        """
        Parameters:
        
        - block: block to be constructed (for ResNet50: Bottleneck)
        
        - layers: no of conv layers in each block, passed as a list (for ResNet50: [3 4, 6, 3])
        
        - num_classes(default: 3): no. of classes
        
        - base(default: True): specifies the input dimension of the image, it is true for 64 x 64 x 64 images and false for others
        
        """
        
        self.base = base
        
        if not self.base:
            self.in_channels = 64
            self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm3d(64)

        else:
            self.in_channels = 128
            self.conv1 = nn.Conv3d(1, 64, kernel_size = 3, stride=2)
            self.bn1 = nn.BatchNorm3d(64)
            self.conv2 = nn.Conv3d(64, 64, kernel_size = 3)
            self.bn2 = nn.BatchNorm3d(64)
            self.conv3 = nn.Conv3d(64, 128, kernel_size = 3)
            self.bn3 = nn.BatchNorm3d(128)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    """
    function for building convolution blocks
    
    """

    def _make_layer(self, block, out_channels, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * 4),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * 4
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        p1 = self.maxpool(x)

        p2 = self.layer1(p1)
        p3 = self.layer2(p2)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)

        return p4, p5

class PPM(nn.Module):
    
    def __init__(self, in_dimension, reduction_dimension, ppm_layers=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        
        """
        Parameters:
        
        - in_dimension: no. of input channels
        
        - reduction_dimension: no. of output channels
        
        - ppm_layers(default: (1, 2, 3, 6)): output dimension of the features after adaptive average pooling, passed as a list
        
        """
        
        self.features = []
        self.ppm_layers = ppm_layers

        for ppm_layer in ppm_layers:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(ppm_layer),
                nn.Conv3d(in_dimension, in_dimension, kernel_size=1, bias=False)
            ))

        self.features = nn.ModuleList(self.features)
        self.conv = nn.Conv3d(in_dimension * (len(self.ppm_layers) + 1), reduction_dimension, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.size()
        result = [x]

        for feature in self.features:
            result.append(feature(x))

        ppm_feats = [F.upsample(x, size=(d, h, w), mode='trilinear') for x in result] 
        result = self.conv(torch.cat(ppm_feats, 1))   
        return result

class PSPNet(nn.Module):
    
    """
    Generate Model Architecture
    """
    
    def __init__(self, n_classes=4, zoom_factor=0, sizes=(1, 2, 3, 6), base=True):

        super().__init__()
        
        """
        Parameters:
        
        - n_classes(deafult :4): no. of classes
        
        - zoom_factor(default: 0): the factor by which the output has to be downsampled
        
        - ppm_layers(default: (1, 2, 3, 6)): a list of output dimension of the features after adaptive average pooling, to be passed in the PPM
        
        """

        self.zoom_factor = zoom_factor
        self.in_dimension = 2048
        self.base = base
        
        """
        define the encoder
        
        """
        
        self.encoder = ResNet50_3D(Bottleneck, [3, 4, 6, 3]) #specify base=False if input dim = 64
        
        """
        define the pyramid pooling network
        
        """

        self.ppm = PPM(self.in_dimension, 1024, sizes)

        self.dropout_1 = nn.Dropout3d(p=0.3)

        self.dropout_2 = nn.Dropout3d(p=0.15)

        self.conv1 = nn.Sequential(
            nn.Conv3d(1024, 512, 1, padding=1),
            nn.BatchNorm3d(512),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(512, 128, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU()
        )

        self.classifier = nn.Sequential(
            nn.Conv3d(128, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
      x_size = x.size()

      d = x_size[2]
      h = x_size[3]
      w = x_size[4]

      aux, main = self.encoder(x)
      main = self.ppm(main)

      main = self.dropout_1(main)

      main = self.conv1(main) 
      main = self.dropout_2(main)

      main = self.conv2(main) 
      main = self.dropout_2(main)

      main = F.upsample(main, size=(d, h, w), mode='trilinear')
      main = self.classifier(main)

      return main

import torch
import torch.nn as nn

from torch.autograd import Variable

class BasicBlock(nn.Module):
  
    """
    Class for each Basic Block in ResNet18
    
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
      
        """
        Parameters:
        
        - in_planes: input channels of image passed to block
        
        - out_planes: output channels required
        
        - stride(default: 1): stride to be assigned
        
        - kernel_size: kernel size for conv in each block
        
        - stride(default: 1): stride to be assigned to each block
        
        - padding(default: 0): amount of padding to be assigned to each block
        
        - groups(default: 1): groups to be assigned to each block
        
        - bias(default: False): boolean bias to be assigned to each block
        
        """
      
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.downsample = None
        
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
      
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #if self.downsample is not None:
        #   residual = self.downsample(x)

        #print('out: ',out.size(), residual.size())
        #out += residual
        
        out = self.relu(out)

        return out

class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
      
        """
        Parameters:
        
        - in_planes: no. of input channels
        
        - out_planes: no. of output channels
        
        - kernel_size: kernel size for conv in each block
        
        - stride(default: 1): stride to be assigned to each block
        
        - padding(default: 0): amount of padding to be assigned to each block
        
        - groups(default: 1): groups to be assigned to each block
        
        - bias(default: False): boolean bias to be assigned to each block
        
        """
      
        super(Encoder, self).__init__()
        
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
      
        x = self.block1(x)
        x = self.block2(x)

        return x

class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        
        """
        Parameters:
        
        - in_planes: no. of input channels
        
        - out_planes: no. of output channels
        
        - kernel_size: kernel size for conv transpose
        
        - stride(default: 1): stride to be assigned to conv transpose
        
        - padding(default: 0): amount of padding to be assigned to conv transpose
        
        - output_padding(default: 0): output padding to be assigned to conv transpose
        
        - bias(default: False): boolean bias to be assigned to each block
        
        """
        
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv3d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm3d(in_planes//4),
                                nn.ReLU(inplace=True),)
        
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode = 'trilinear'),
                                nn.BatchNorm3d(in_planes//4),
                                nn.ReLU(inplace=True),)
        
        self.conv2 = nn.Sequential(nn.Conv3d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm3d(out_planes),
                                nn.ReLU(inplace=True),)
        
    def forward(self, x):
      
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)

        return x

class LinkNet(nn.Module):

    """
    Generate Model Architecture
    """

    def __init__(self, n_classes = 4):

        """
        Parameters:
        
        n_classes(default: 4): number of output neurons
        
        """

        super(LinkNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 1, 1, 0)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 1, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 1, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 1, 2, 1)


        # Classifier
        self.conv2 = nn.Conv3d(64, 32, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)       
        self.conv3 = nn.Conv3d(32, n_classes, 3, 1, 1)
        self.softmax = nn.Softmax()


    def forward(self, x):
      
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = self.decoder1(d2)
        
        # Classifier
        y = self.conv2(d1)
        y = self.up(y)
        y = self.conv3(y)

        y = self.softmax(y)

        return y

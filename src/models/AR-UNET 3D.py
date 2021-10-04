class Block(nn.Module):
    def __init__(self,in_channels,out_channels,padding,stride=1):
        super(Block,self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=padding)
                                 
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
                                
        
        self.skip = nn.Sequential(nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
                                 nn.BatchNorm3d(out_channels))
        
        
    def forward(self,x):
        inp = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        #print('block',x.size())

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        #print('block',x.size())

        skip = self.skip(inp)
        
        x = x + skip

        return x
      
class Attention(nn.Module):
    def __init__(self,fx,fg,final):
        super(Attention,self).__init__()
        self.convg = nn.Conv3d(fg,final,kernel_size=1,stride=1,padding=0,bias=True)
        self.bng = nn.BatchNorm3d(final)
        
        self.convx = nn.Conv3d(fx,final,kernel_size=1,stride=1,padding=0,bias=True)
                                
        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(nn.Conv3d(final,1,kernel_size=1,stride=1,padding=0,bias=True),
                                nn.BatchNorm3d(1),
                                nn.Sigmoid())
    
    def forward(self,x,g):
        g = self.convg(g)
        g = self.bng(g)
        x = self.convx(x)
        x = self.bng(x)
        r = self.relu(g+x)

        psi = self.psi(r)
        psi = x*psi

        return psi
      
class AttnResUnet3D(nn.Module):
    def __init__(self,Block,Attn,img_channels,channels):
        super(AttnResUnet3D,self).__init__()

        #Encoder part
        self.input = nn.Sequential(nn.Conv3d(img_channels,channels[0],kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm3d(channels[0]),
                                   nn.ReLU(),
                                   nn.Conv3d(channels[0],channels[0],kernel_size=3,stride=1,padding=1)) #makes it 64x64
        
        self.skip1 = nn.Conv3d(img_channels,channels[0],kernel_size=3,stride=1,padding=1) 

        self.b1 = Block(channels[0],channels[1],1,2)
        self.b2 = Block(channels[1],channels[2],1,2)

        #Bridge
        self.bridge = Block(channels[2],channels[3],1,2)

        #Decoder part

        self.up1 = nn.Upsample(scale_factor=2,mode='trilinear')
        self.attn1 = Attn(channels[3],channels[3],channels[3])
        self.b3 = Block(channels[3]+channels[2],channels[2],1,1)

        self.up2 = nn.Upsample(scale_factor=2,mode='trilinear')
        self.attn2 = Attn(channels[2],channels[2],channels[2])
        self.b4 = Block(channels[2]+channels[1],channels[1],1,1)

        self.up3 = nn.Upsample(scale_factor=2,mode='trilinear')
        self.attn3 = Attn(channels[1],channels[1],channels[1])
        self.b5 = Block(channels[1]+channels[0],channels[0],1,1)
    
        #Final block
        self.final = nn.Conv3d(channels[0],4,1,1)
        
        self.soft = nn.Softmax(dim=1)

        self.resize1 = nn.Sequential(nn.Conv3d(64,128,kernel_size=1,stride=1),
                             nn.BatchNorm3d(128),
                             nn.ReLU())
        self.resize2 = nn.Sequential(nn.Conv3d(128,256,kernel_size=1,stride=1),
                             nn.BatchNorm3d(256),
                             nn.ReLU())
        self.resize3 = nn.Sequential(nn.Conv3d(256,512,kernel_size=1,stride=1),
                             nn.BatchNorm3d(512),
                             nn.ReLU())
        
    
        for x in self.modules():
            if isinstance(x, nn.Conv3d):
                nn.init.kaiming_normal_(x.weight,nonlinearity='relu',mode='fan_out')
            elif isinstance(x, nn.BatchNorm3d):
                nn.init.constant_(x.weight,1)
                nn.init.constant_(x.bias,0)
        

    def forward(self,x):
       
        x1 = self.input(x) + self.skip1(x) #64
        atx1 = self.resize1(x1)

        x2 = self.b1(x1) #128
        atx2 = self.resize2(x2)

        x3 = self.b2(x2) #256
        atx3 = self.resize3(x3)

        x4 = self.bridge(x3)
        
        x4 = self.up1(x4)

        x4 = self.attn1(atx3,x4)
        x5 = torch.cat([x4,x3],dim=1)

        x6 = self.b3(x5)
        x6 = self.up2(x6)

        x6 = self.attn2(atx2,x6)
        x7 = torch.cat([x6,x2],dim=1)

        x8 = self.b4(x7)
        x8 = self.up3(x8)

        x8 = self.attn3(atx1,x8)
        x9 = torch.cat([x8,x1],dim=1)

        x9 = self.b5(x9)
        x_out = self.final(x9)
        x_out = self.soft(x_out)

        return x_out
     

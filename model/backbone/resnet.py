import torch
import numpy as np
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self,inch,outch,ksize,stride,padding):
        super(ConvBlock,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(inch,outch,ksize,stride,padding),
            nn.BatchNorm2d(outch),
            nn.ReLU
        )

    def forward(self,x):
        x=self.block(x)
        return x
class BasicBlock(nn.Module):
    def __init__(self,inch,outch,ksize,striding,padding):
        super(BasicBlock,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(inch,outch,ksize,striding,padding,bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU(),
            nn.Conv2d(outch,outch,ksize,1,1,bias=False),
            nn.BatchNorm2d(outch)
        )
        self.flag=0
        if inch!=outch:
            self.flag=1
            self.downsample=DownSample(inch,outch,1,2,0)
    def forward(self,x):
        return self.block(x)+x if self.flag==0 else self.block(x)+self.downsample(x)

class DownSample(nn.Module):
    def __init__(self,inch,outch,ksize,stride,padding):
        super(DownSample,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(inch,outch,ksize,stride,padding),
            nn.BatchNorm2d(outch)
        )
    def forward(self, x):
        return self.block(x)
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1=ConvBlock(3,32,3,2,0)
        self.maxpool=nn.MaxPool2d(3,2,1,1,ceil_mode=False)
        self.layer1=nn.Sequential(
            BasicBlock(64,64,3,1,1),
            BasicBlock(64,64,3,1,1)
        )
        self.layer2=nn.Sequential(
            BasicBlock(64,128,3,2,1),
            BasicBlock(128,128,3,1,1)
        )
        self.layer3=nn.Sequential(
            BasicBlock(128,256,3,2,1),
            BasicBlock(256,256,3,1,1)
        )
        self.layer4=nn.Sequential(
            BasicBlock(256,512,3,2,1),
            BasicBlock(512, 512, 3, 1, 1)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool(x)
        fea1=self.layer1(x)
        fea2=self.layer2(fea1)
        fea3=self.layer3(fea2)
        fea4=self.layer4(fea3)
        return fea1,fea2,fea3,fea4
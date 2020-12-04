import torch
import torch.nn as nn
class Upsample():
    def __init__(self,inch,outch):
        super(Upsample,self).__init__()
        self.block=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inch,outch,3,1,1)
        )

    def forward(self,x):
        x=self.block(x)
        return x

class FPN(nn.Module):
    def __init__(self):
        super(FPN,self).__init__()

    def forward(self,*arg):
        pari=[arg[0]]
        for fea in arg[1:]:
            fea_up=Upsample(pari[-1].shape[1],fea.shape[1]).forward(pari[-1])
            pari.append(fea_up+fea)
        return pari


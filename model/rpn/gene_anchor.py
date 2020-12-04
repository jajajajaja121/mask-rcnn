import torch
import torch.nn as nn
import numpy as np

class AnchorGenerator():
    def __init__(self,size,ratio,fea):
        self.size=size
        self.ratio=ratio
        self.all_anchor_scale=self._get_all_anchor_scale()
        self.fea_h,self.fea_w=fea.shape[2],fea.shape[3]
        all_anchor=self._get_all_anchor()
    def _get_all_anchor_scale(self):
        all_anchor=[]
        for size in self.size:
            area=size**2
            for ratio in self.ratio:
                piece=np.sqrt(area/(ratio[0]*ratio[1]))
                all_anchor.append((piece*ratio[0],piece*ratio[1]))
        return all_anchor

    def _get_all_anchor(self):
        all_anchor=torch.zeros(self.fea_w,self.fea_h,len(self.all_anchor_scale),4)
        for h in range(self.fea_h):
            for w in range(self.fea_w):
                for k,anchor_w,anchor_h in enumerate(self.all_anchor_scale):
                    anchor=(h-anchor_h,w-anchor_w,h+anchor_h,w+anchor_w)
                    all_anchor[h,w,k,:]=anchor
        all_anchor=self._check_anchor_out_board(all_anchor)
        return all_anchor

    def _check_anchor_out_board(self,anchors):
        anchors=torch.clamp(anchors[:,:,:,0],0,self.fea_h)
        anchors=torch.clamp(anchors[:,:,:,1],0,self.fea_w)
        anchors=torch.clamp(anchors[:,:,:,2],0,self.fea_h)
        anchors=torch.clamp(anchors[:,:,:,3],0,self.fea_w)
        return anchors.view(-1,4)

class AnchorTarget():
    def __init__(self,anchors,target,stride):
        self.anchors=anchors*stride
        self.ground_truth=target

    def cal_iou(self,result,target):
        iou=torch.zeros((result.shape[0],target.shape[0]))








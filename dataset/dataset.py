import torch.utils.datasets as datasets
import torch
import os
import numpy as np
import os.path as osp
from PIL import Image
import json

class MaskDataset(datasets):
    def __init__(self,img_path,ann_path,mode,transform):
        super(MaskDataset,self).__init__()
        coco_label=json.load(open(ann_path))
        self.image_list=coco_label['images']
        self.ann_list=coco_label['annotations']
        self.categories=coco_label['categories']
        self.img_path=img_path
        self.transform=transform
        self.img_label_dict=self._create_img_label_dict()
    def __getitem__(self, index):
        im_info=self.img_list(index)
        im_path=osp.join(self.img_path,im_info['file_name'])
        image=Image.open(im_path)
        image=self.transform(image)
        label=self.img_label_dict[im_info['id']]
        return im_info,image,label

    def _create_img_label_dict(self):
        img_label_dict={}
        for img_info in self.image_list:
            id=img_info['id']
            img_label_dict[id]=[]
            for ann_info in self.ann_list:
                if ann_info['image_id']==id:
                    img_label_dict[id].append(ann_info)
        return img_label_dict
    def __len__(self):
        return len(self.image_list)

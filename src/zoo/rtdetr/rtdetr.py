"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'SR','Fusion', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        SR: nn.Module, 
        Fusion: nn.Module,
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.SR = SR
        self.Fusion = Fusion
        self.encoder = encoder
        self.decoder = decoder
       
        
    def forward(self, x, targets=None):
      
        x,x_image = self.backbone(x)
        # print(x[0].shape,x[1].shape,x[2].shape,x[3].shape,x_image.shape)
        x1,Super_feature = self.SR(x_image) #下采样后图片放入进行特征提取
        # print(Super_feature[0].shape,Super_feature[1].shape,Super_feature[2].shape)
        Super_features=[Super_feature[1],Super_feature[2]]
        # print(Super_features[0].shape,Super_features[1].shape)
        x_Fusion =  self.Fusion(Super_features,x)
        # print(x_Fusion[0].shape,x_Fusion[1].shape,x[3].shape)


        x=[x_Fusion[0],x_Fusion[1],x[3]]
        # x=[x[1],x[2],x[3]]
        
        # print(x[0].shape,x[1].shape,x[2].shape)
        x = self.encoder(x)    
        # print('encoder',len(x),x[0].shape,x[1].shape,x[2].shape)    
        x = self.decoder(x, targets)

        return x,x1
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 

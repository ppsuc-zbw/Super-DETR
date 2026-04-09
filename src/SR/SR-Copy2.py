"""Copyright(c) 2024 zhengbowen. All Rights Reserved.
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict
from skimage.io import imsave
# from common import get_activation, FrozenBatchNorm2d
# from core import register

from ..nn import get_activation, FrozenBatchNorm2d

from ..core import register


class ConvLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act) 

    def forward(self, input):
        out=self.act(self.norm(self.conv(input)))
        return out



class Conv_upsample(nn.Module):

    def __init__(self, ch_in, stride, act='relu'):
        super().__init__()



        self.conv1 = ConvLayer(ch_in, ch_in, 3, stride, act=act)
        self.conv2 = ConvLayer(ch_in, ch_in//2, 3, stride, act=act)
        self.act = nn.Identity() if act is None else get_activation(act) 
        self.super1 = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, input):
        out = self.conv1(input)
        out = self.super1(out)
        # print(out.shape)
        # out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv2(out)
        # out = self.act(out)

        return out

class Blocks_up(nn.Module):
    def __init__(self, block, ch_in, count, act='relu'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):

            self.blocks.append(
                block(
                    ch_in, 
                    stride=1, 
                    act=act)
            )
            ch_in=ch_in//2
    def forward(self, input):
        out = input
        # print(input.shape)
        for block in self.blocks:
            out = block(out)
        return out



class Bottle(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, act='relu'):
        super().__init__()

        # self.branch2a = ConvLayer(ch_in, ch_in, 3, stride, act=act)
        # self.super1 = nn.PixelShuffle(2)
        self.super1  = nn.Upsample(scale_factor=2, mode='nearest')
        self.branch2b = ConvLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2c = ConvLayer(ch_out, ch_out, 3, stride, act=act)
      

    def forward(self, input):
        # out = self.branch2a(input)
        out = self.super1(input)

        out = self.branch2b(out)
        out = self.branch2c(out)

        return out



class Blocks_s(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, act='relu', ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in, 
                    ch_out,
                    stride=1, 
                    act=act)
            )
    def forward(self, input):
        out = input
        for block in self.blocks:
            out = block(out)
        return out



class ASFF(nn.Module):
    def __init__(self, Dimlist):
        super().__init__()

        self.dim = Dimlist
        self.Upsample_Layers = nn.ModuleList()
        num_stages= 4
        block= Conv_upsample
        act='relu'

        
        for i in range(num_stages):
            
            self.Upsample_Layers.append(
                Blocks_up(Conv_upsample, Dimlist[i], i+1, act=act)
            )

        compress_c = 8 

        self.weight_level_0 = ConvLayer(self.dim[0]//2, compress_c, 1, 1)
        self.weight_level_1 = ConvLayer(self.dim[0]//2, compress_c, 1, 1)
        self.weight_level_2 = ConvLayer(self.dim[0]//2, compress_c, 1, 1)
        self.weight_level_3 = ConvLayer(self.dim[0]//2, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.decode1 = ConvLayer(compress_c*4, compress_c*2, 3, 1,act='relu')
        self.decode2 = ConvLayer(compress_c*2, compress_c, 3, 1,act='relu')
        self.decode3 = ConvLayer(compress_c, 3, 3, 1,act='relu')

    def forward(self, MutliScale_Feature):

        Super_feature_list=[]
        for idx, stage in enumerate(self.Upsample_Layers):
            ###通过stage将所有特征映射到了同一个尺寸
            # print(MutliScale_Feature[idx].shape)
            feature = stage(MutliScale_Feature[idx])
            Super_feature_list.append(feature)
            
        # print(Super_feature_list[0].shape,Super_feature_list[1].shape,Super_feature_list[2].shape,Super_feature_list[3].shape)
        level_0_weight_v = self.weight_level_0(Super_feature_list[0])
        level_1_weight_v = self.weight_level_1(Super_feature_list[1])
        level_2_weight_v = self.weight_level_2(Super_feature_list[2])
        level_3_weight_v = self.weight_level_3(Super_feature_list[3])
        
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v,level_3_weight_v),1)

        
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
     
        fused_out_reduced = Super_feature_list[0] * levels_weight[:,0:1,:,:]+\
                            Super_feature_list[1] * levels_weight[:,1:2,:,:]+\
                            Super_feature_list[2] * levels_weight[:,2:3,:,:]+\
                            Super_feature_list[3] * levels_weight[:,3:4,:,:]
        
        # print(fused_out_reduced.shape,'fused_out')
        out = self.decode1(fused_out_reduced)
        out = self.decode2(out)
        out = self.decode3(out)
        
        outputs_save=(out*255).cpu().detach().numpy().astype('uint8')
        # outputs_save=outputs_save

        # print(outputs_save[0:1,:, :, :].squeeze().shape)
        outputs_save=outputs_save[0:1,:, :, :].squeeze()
        outputs_save=outputs_save.transpose(1,2,0)
        # print(outputs_save.max(),outputs_save.min())
        imsave('/root/autodl-tmp/Super_re/' +'Spuer.png',outputs_save)
        return out,MutliScale_Feature



@register()
class SR(nn.Module):
    def __init__(
        self, 
        num_stages=4, 
        act='relu',
        block_nums = 1,
        freeze_norm=True, 
        pretrained=False
         ):
        super().__init__()
        
        
        block = Bottle 

        ###该层为psrest输出的网络特征数
        ch_PSout_list = [64 ,128, 256, 512]
        ###该层为输入到超分辨率网络中的网络特征通道数
        ch_in_list = [block.expansion * v for v in ch_PSout_list]
        ###通道数依次缩减，分辨率不断提高，最终恢复成 64 ，128 ，256， 512 通道数
        ch_out_list = [64 ,128, 256, 512]
        # ch_out_list =  [32 ,64, 128, 256]
        
        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            self.res_layers.append(
                Blocks_s(block, ch_in_list[i], ch_out_list[i], block_nums, act=act)
            )
        self.ASFF= ASFF(ch_PSout_list)

        
        if freeze_norm:
            self._freeze_norm(self)
    
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        
        outs = []
        for idx, stage in enumerate(self.res_layers):
            input=x
           
            feature = stage(input[idx])
            ####查看返回的张量维度大小
            # print(input.shape,"!!!")
 
            outs.append(feature)
        # print(outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape)

        out,MutliScale_Feature = self.ASFF(outs)
        
        ###此时outs 是一个列表，列表内排列矩阵torch.Size([1, 64, 336, 336]), torch.Size([1, 128, 168, 168]), torch.Size([1, 256, 84, 84]), torch.Size([1, 512, 42, 42])
        ### out 是一个重建的超分辨率图像
        return out,MutliScale_Feature


if __name__ == '__main__':

    m = SR()
    data0 = torch.randn(1,256 , 168, 168)
    data1 = torch.randn(1,512 , 84, 84)
    data2 = torch.randn(1,1024 , 42, 42)
    data3 = torch.randn(1,2048 , 21, 21)
    out=[data0, data1,data2,data3]
    output,MutliScale_Feature = m(out)
    # print([o.shape for o in MutliScale_Feature])

    # output[0].mean().backward()
"""Copyright(c) 2024 zhengbowen. All Rights Reserved.
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict
from skimage.io import imsave
from common import get_activation, FrozenBatchNorm2d
from core import register

# from ..nn import get_activation, FrozenBatchNorm2d

# from ..core import register


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
        
        # 需求分析：
        # 输入: [1, 64, 200, 200]
        # 输出1: [1, 64, 400, 400]  -> 上采样2倍，通道不变
        # 输出2: [1, 128, 200, 200] -> 分辨率不变，通道加倍
        # 输出3: [1, 256, 100, 100] -> 下采样2倍，通道加倍
        # 输出4: [1, 512, 50, 50]   -> 下采样2倍，通道加倍
        
        # 定义第一层：上采样模块 (利用原有的Bottle结构，它包含Upsample)
        # Bottle(64, 64) 会将输入 upsample 2x，并保持通道为64
        self.layer1 = Bottle(64, 64, stride=1, act=act)
        
        # 定义第二层：通道调整模块 (分辨率不变)
        self.layer2 = ConvLayer(64, 128, kernel_size=3, stride=1, act=act)
        
        # 定义第三层：下采样模块 (stride=2)
        self.layer3 = ConvLayer(128, 256, kernel_size=3, stride=2, act=act)
        
        # 定义第四层：下采样模块 (stride=2)
        self.layer4 = ConvLayer(256, 512, kernel_size=3, stride=2, act=act)

        # ASFF 模块，传入各阶段的通道数列表 [64, 128, 256, 512]
        self.ASFF = ASFF([64, 128, 256, 512])

        
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
        # x: [1, 64, 200, 200]
        
        # 阶段1: 上采样 -> [1, 64, 400, 400]
        out1 = self.layer1(x)
        
        # 阶段2: 通道变换 -> [1, 128, 200, 200]
        out2 = self.layer2(x)
        
        # 阶段3: 下采样 -> [1, 256, 100, 100]
        out3 = self.layer3(out2)
        
        # 阶段4: 下采样 -> [1, 512, 50, 50]
        out4 = self.layer4(out3)
        
        # 收集多尺度特征
        outs = [out1, out2, out3, out4]
        
        # 输入到 ASFF 模块进行特征融合与重建
        out, MutliScale_Feature = self.ASFF(outs)
        
        return out, MutliScale_Feature

from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count

# 假设 SR 类已经在当前环境中定义或导入
# from your_script import SR 

def count_model_params_flops():
    # 1. 实例化模型
    model = SR()
    
    # 2. 切换到评估模式 (这很重要，因为 BatchNorm 在 train 和 eval 模式下计算逻辑不同)
    model.eval()
    
    # 3. 创建符合输入要求的 dummy input
    # 输入形状: [1, 64, 200, 200]
    input_data = torch.randn(1, 64, 200, 200)

    # print("="*30)
    print("方法一：使用 thop 库 (常用)")
    # print("="*30)
    
    # 使用 thop.profile 计算 MACs (Multiply-Add Operations) 和 Params
    # thop 输出的是 MACs，通常 FLOPs ≈ 2 * MACs
    macs, params = profile(model, inputs=(input_data, ), verbose=False)
    # print(macs, params)
    # 格式化输出 (自动转换为 M 或 G 单位)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
    
    print(f"Total Parameters: {params_formatted}")
    print(f"Total MACs: {macs_formatted} (FLOPs 约为 {float(macs_formatted.strip('G ').strip('M ')) * 2:.3f} {macs_formatted[-2:]})")

    # print("\n" + "="*30)
    # print("方法二：使用 fvcore 库 (Facebook官方，更准确)")
    # print("="*30)
    
    # # 计算 FLOPs
    # flops_counter = FlopCountAnalysis(model, input_data)
    # total_flops = flops_counter.total()
    
    # # 计算参数量
    # param_counter = parameter_count(model)
    # total_params = param_counter[""]
    
    # # 格式化输出
    # if total_flops > 1e9:
    #     flops_str = f"{total_flops / 1e9:.3f} G"
    # else:
    #     flops_str = f"{total_flops / 1e6:.3f} M"
        
    # print(f"Total Parameters: {total_params / 1e6:.3f} M")
    # print(f"Total FLOPs: {flops_str}")
    
    # 打印各层详细信息 (可选)
    # print(flops_counter.by_operator())


if __name__ == '__main__':

    m = SR()
    data0 = torch.randn(1,64 , 168, 168)
    # data1 = torch.randn(1,512 , 84, 84)
    # data2 = torch.randn(1,1024 , 42, 42)
    # data3 = torch.randn(1,2048 , 21, 21)
    # out=[data0, data1,data2,data3]
    output,MutliScale_Feature = m(data0)
    print([o.shape for o in MutliScale_Feature])
    count_model_params_flops()

    # output[0].mean().backward()
"""Copyright(c) 2024 zhengbowen. All Rights Reserved.
"""


from ..nn import get_activation, FrozenBatchNorm2d

from ..core import register



from skimage.io import imsave
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict
# from common import get_activation, FrozenBatchNorm2d
# from core import register


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
        out = self.act(self.norm(self.conv(input)))
        return out

class Conv_upsample(nn.Module):
    def __init__(self, ch_in, stride, act='relu'):
        super().__init__()
        # 优化：先压缩通道，再上采样
        self.conv1 = ConvLayer(ch_in, ch_in//2, 3, stride, act=act)
        self.super1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = ConvLayer(ch_in//2, ch_in//2, 3, stride, act=act)
        
    def forward(self, input):
        out = self.conv1(input)
        out = self.super1(out)
        out = self.conv2(out)
        return out

class Blocks_up(nn.Module):
    def __init__(self, block, ch_in, count, act='relu'):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(ch_in, stride=1, act=act)
            )
            ch_in = ch_in // 2
            
    def forward(self, input):
        out = input
        for block in self.blocks:
            out = block(out)
        return out

class DownSample(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.conv = ConvLayer(ch_in, ch_out, 3, stride=2, act=act)
        
    def forward(self, x):
        return self.conv(x)

class ASFF(nn.Module):
    def __init__(self, Dimlist):
        super().__init__()
        self.dim = Dimlist
        self.Upsample_Layers = nn.ModuleList()
        num_stages = 3  # 修改：处理3个层级
        block = Conv_upsample
        act = 'relu'
        
        # 目标输出尺寸：640x640
        # 输入特征尺寸: 320, 160, 80
        # 需要的上采样倍数:
        # Stage 0: 320 -> 640 (x2)  -> 需要 1 个 UpBlock
        # Stage 1: 160 -> 640 (x4)  -> 需要 2 个 UpBlock
        # Stage 2: 80 -> 640 (x8)   -> 需要 3 个 UpBlock
        upsample_counts = [1, 2, 3]
        
        for i in range(num_stages):
            self.Upsample_Layers.append(
                Blocks_up(Conv_upsample, Dimlist[i], upsample_counts[i], act=act)
            )

        # 计算最终融合通道数
        # Stage 0 (64) -> count 1 -> 64/2 = 32
        # Stage 1 (128) -> count 2 -> 128/4 = 32
        # Stage 2 (256) -> count 3 -> 256/8 = 32
        final_ch = 32
        align_ch = 32
        
        compress_c = 8 
        
        # 权重生成层 (修改：仅需3个层级)
        self.weight_level_0 = ConvLayer(align_ch, compress_c, 1, 1)
        self.weight_level_1 = ConvLayer(align_ch, compress_c, 1, 1)
        self.weight_level_2 = ConvLayer(align_ch, compress_c, 1, 1)

        # 修改：输出通道数为3
        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        
        # 解码层
        self.decode1 = ConvLayer(align_ch, compress_c*2, 3, 1, act='relu')
        self.decode2 = ConvLayer(compress_c*2, compress_c, 3, 1, act='relu')
        self.decode3 = ConvLayer(compress_c, 3, 3, 1, act='relu')

    def forward(self, MutliScale_Feature):
        Super_feature_list = []
        for idx, stage in enumerate(self.Upsample_Layers):
            feature = stage(MutliScale_Feature[idx])
            Super_feature_list.append(feature)
            
        # 此时所有特征均为 [Batch, 32, 640, 640]
        
        level_0_weight_v = self.weight_level_0(Super_feature_list[0])
        level_1_weight_v = self.weight_level_1(Super_feature_list[1])
        level_2_weight_v = self.weight_level_2(Super_feature_list[2])
        
        # 修改：concat 3个层级
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
     
        # 修改：融合3个层级
        fused_out_reduced = Super_feature_list[0] * levels_weight[:,0:1,:,:] + \
                            Super_feature_list[1] * levels_weight[:,1:2,:,:] + \
                            Super_feature_list[2] * levels_weight[:,2:3,:,:]
        
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
        return out, MutliScale_Feature
class FeatureProcessor(nn.Module):
    def __init__(self, channels, act='relu'):
        super().__init__()
        self.process_layers = nn.ModuleList()
        # 对每个尺度进行 stride=2 的卷积
        for ch in channels:
            self.process_layers.append(
                ConvLayer(ch, ch, 3, stride=2, act=act)
            )
    
    def forward(self, features):
        processed_features = []
        for feat, layer in zip(features, self.process_layers):
            processed_features.append(layer(feat))
        return processed_features
@register()
class SR(nn.Module):
    def __init__(
        self, 
        num_stages=3, # 修改：层级数改为3
        act='relu',
        block_nums = 1,
        freeze_norm=True, 
        pretrained=False
         ):
        super().__init__()
        
        # 输入: [B, 64, 320, 320]
        # 需要生成的金字塔: [64, 320], [128, 160], [256, 80]
        
        # Layer 0: 保持 320x320, 通道 64
        self.layer0 = ConvLayer(64, 64, 3, 1, act=act)
        
        # Layer 1: 下采样 -> 160, 通道 128
        self.layer1 = DownSample(64, 128, act=act)
        
        # Layer 2: 下采样 -> 80, 通道 256
        self.layer2 = DownSample(128, 256, act=act)
        
        # 移除了 Layer 3
            
        self.ASFF = ASFF([64, 128, 256])
        intermediate_channels = [64, 128, 256, 512]
        self.processor = FeatureProcessor(intermediate_channels, act=act)
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
        """
        x: Tensor [B, 64, 320, 320]
        """
        if isinstance(x, (list, tuple)):
            x = x[0]
            
        outs = []
        # 生成特征金字塔
        out0 = self.layer0(x)       # [1, 64, 320, 320]
        outs.append(out0)
        
        out1 = self.layer1(out0)    # [1, 128, 160, 160]
        outs.append(out1)
        
        out2 = self.layer2(out1)    # [1, 256, 80, 80]
        outs.append(out2)
            
        # ASFF融合并上采样到 640x640
        out, MutliScale_Feature = self.ASFF(outs)
        MutliScale_Feature=self.processor(MutliScale_Feature)
        return out, MutliScale_Feature



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


if __name__ == '__main__':
    # 测试代码
    model = SR()
    model.eval()
    
    input_data = torch.randn(1, 64, 320, 320)
    
    from thop import profile, clever_format
    macs, params = profile(model, inputs=(input_data,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    
    with torch.no_grad():
        out, features = model(input_data)
        
    print("SR模块生成的特征金字塔:")
    for i, f in enumerate(features):
        print(f"  Stage {i}: {f.shape}")
        
    print(f"\n最终输出尺寸: {out.shape}") # 预期: [1, 3, 640, 640]
    print(f"参数量: {params}")
    print(f"计算量: {macs}")



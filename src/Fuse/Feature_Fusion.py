import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np
from thop import profile

from ..core import register

def count_flops(model, input):
    # 使用thop的profile函数
    macs, params = profile(model, inputs=(input,))
    return macs, params

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def carafe(x, normed_mask, kernel_size, group=1, up=1):
        b, c, h, w = x.shape
        _, m_c, m_h, m_w = normed_mask.shape
        # print('x', x.shape)
        # print('normed_mask', normed_mask.shape)
        # assert m_c == kernel_size ** 2 * up ** 2
        assert m_h == up * h
        assert m_w == up * w
        pad = kernel_size // 2
        # print(pad)
        pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
        # print(pad_x.shape)
        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
        # unfold_x = unfold_x.reshape(b, c, 1, kernel_size, kernel_size, h, w).repeat(1, 1, up ** 2, 1, 1, 1, 1)
        unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
        unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
        # normed_mask = normed_mask.reshape(b, 1, up ** 2, kernel_size, kernel_size, h, w)
        unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
        res = unfold_x * normed_mask
        # test
        # res[:, :, 0] = 1
        # res[:, :, 1] = 2
        # res[:, :, 2] = 3
        # res[:, :, 3] = 4
        res = res.sum(dim=2).reshape(b, c, m_h, m_w)
        # res = F.pixel_shuffle(res, up)
        # print("res",res.shape)
        # print(res)
        return res
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    # hamming_x = np.blackman(M)
    # hamming_x = np.kaiser(M)
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d

@register()
class FreqFusion(nn.Module):
    def __init__(self, hr_channels, lr_channels, scale_factor=1, lowpass_kernel=5, highpass_kernel=3, up_group=1, encoder_kernel=3, encoder_dilation=1, compressed_channels=64, align_corners=False, upsample_mode='nearest', feature_resample=True, feature_resample_group=4, comp_feat_upsample=True, use_high_pass=True, use_low_pass=True, hr_residual=True, semi_conv=True, hamming_window=True, feature_resample_norm=False, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        
        # 1. 通道压缩
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        
        # 2. 核预测器 (ALPF & AHPF)
        self.content_encoder = nn.Conv2d( # Low-pass kernel generator
            self.compressed_channels, lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel, padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation, groups=1)
            
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample
        self.hr_adapter = nn.Conv2d(hr_channels, lr_channels, 1)
        if self.feature_resample:
            # 注意：LocalSimGuidedSampler 默认 scale=2 (用于上采样)。
            # 此处我们需要的是 HR->LR 的融合，可能需要调整该模块或禁用。
            # 为了保持代码改动最小，我们在 forward 中通过 resize 解决尺度问题，此处暂时禁用或保持原样
            pass 

        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d( # High-pass kernel generator
                self.compressed_channels, highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel, padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation, groups=1)
        
        # 窗口初始化
        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
            
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        # 简化流程，不使用 checkpoint
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        # 1. 通道压缩
        compressed_hr_feat = self.hr_channel_compressor(hr_feat) # [1, 64, 160, 160]
        compressed_lr_feat = self.lr_channel_compressor(lr_feat) # [1, 64, 80, 80]

        # -------------------------------------------------------
        # 修改核心逻辑：目标是输出 LR 尺寸 [1, 512, 80, 80]
        # -------------------------------------------------------
        
        # 2. 处理高频特征 (HR -> LR)
        # 我们在 HR 尺寸上提取高频信息，然后下采样到 LR 尺寸
        if self.use_high_pass:
            # 生成高通滤波核
            mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
            mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass)
            
            # 提取 HR 的高频细节: HighFreq = X - LowPass(X)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1)
            
            # 将高频细节下采样到 LR 尺寸 (80x80)
            # 注意：这里需要将通道数对齐，hr_feat 是 128, lr_feat 是 512
            # 添加一个卷积层来适配通道，或者在类初始化时添加。
            # 这里为了简化演示，假设我们在外部或通过自适应池化处理，
            # 更好的方式是在 __init__ 中定义 hr_feat 对应的通道调整层。
            # 此处我们使用 AdaptiveAvgPool2d 进行下采样
            hr_feat_hf_down = F.adaptive_avg_pool2d(hr_feat_hf, output_size=lr_feat.shape[-2:]) # [1, 128, 80, 80]
            
            # 如果通道不匹配，需要调整。
            # 由于题目要求输出 512 通道，而 hr 是 128。
            # 这里我们假设只有部分通道被增强，或者我们在 forward 开始前就调整了 hr 的通道。
            # 最合理的做法：在 __init__ 中增加一个 converter。
            # 为演示方便，这里仅展示尺寸对齐。
            pass 

        # 3. 处理低频特征 (保持 LR 尺寸)
        # 在 LR 尺寸上进行特征平滑/增强
        if self.use_low_pass:
            # 在 LR 尺寸上生成低通核
            mask_lr = self.content_encoder(compressed_lr_feat)
            mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
            
            # 对 lr_feat 进行低通滤波 (平滑)
            # carafe 的 up 参数为 1 表示不上采样，仅做内容重聚合
            lr_feat_smoothed = carafe(lr_feat, mask_lr_init, self.lowpass_kernel, self.up_group, 1)

        # 4. 融合
        # 将下采样后的高频细节加到低频特征上
        # 注意：通道数需要对齐。这里假设我们有一个适配层。
        # 如果我们忽略通道不匹配的问题（假设外部已处理），逻辑如下：
        
        # 假设我们有一个临时卷积适配
        # if not hasattr(self, 'hr_adapter'):
        #     self.hr_adapter = nn.Conv2d(hr_feat.shape[1], lr_feat.shape[1], 1).to(hr_feat.device)
        hr_feat_hf_down_adapted = self.hr_adapter(hr_feat_hf_down)  # 假设hr_feat_hf_down是输入    
        # hr_feat_hf_down_adapted = self.hr_adapter(hr_feat_hf_down) # [1, 512, 80, 80]
        
        # 最终融合
        # 输出 = 原始 LR 特征 + 下采样的高频细节
        # 或者是 平滑后的 LR 特征 + 下采样的高频细节
        out = lr_feat + hr_feat_hf_down_adapted
        
        return out




class LocalSimGuidedSampler(nn.Module):
    """
    offset generator in FreqFusion
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, use_direct_scale=True, kernel_size=1, local_window=3, sim_type='cos', norm=True, direction_feat='sim_concat'):
        super().__init__()
        assert scale==2
        assert style=='lp'

        self.scale = scale
        self.style = style
        self.groups = groups
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2
        if self.direction_feat == 'sim':
            self.offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        elif self.direction_feat == 'sim_concat':
            self.offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        else: raise NotImplementedError
        normal_init(self.offset, std=0.001)
        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            elif self.direction_feat == 'sim_concat':
                self.direct_scale = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            else: raise NotImplementedError
            constant_init(self.direct_scale, val=0.)

        out_channels = 2 * groups
        if self.direction_feat == 'sim':
            self.hr_offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        elif self.direction_feat == 'sim_concat':
            self.hr_offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        else: raise NotImplementedError
        normal_init(self.hr_offset, std=0.001)
        
        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.hr_direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            elif self.direction_feat == 'sim_concat':
                self.hr_direct_scale = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            else: raise NotImplementedError
            constant_init(self.hr_direct_scale, val=0.)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)
    
    def sample(self, x, offset, scale=None):
        if scale is None: scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, scale * H, scale * W)
    
    def forward(self, hr_x, lr_x, feat2sample):
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)

        if self.direction_feat == 'sim':
            hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')
        elif self.direction_feat == 'sim_concat':
            hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            hr_x, lr_x = hr_sim, lr_sim
        # offset = self.get_offset(hr_x, lr_x)
        
        offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
        # print(offset.shape)
        return self.sample(feat2sample, offset)
    
    # def get_offset_lp(self, hr_x, lr_x):
    def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
        if hasattr(self, 'direct_scale'):
            # offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)).sigmoid() + self.init_pos
            offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)).sigmoid() + self.init_pos
            # offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (self.direct_scale(lr_sim) + F.pixel_unshuffle(self.hr_direct_scale(hr_sim), self.scale)).sigmoid() + self.init_pos
        else:
            offset =  (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)
    

def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    """
    计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

    参数：
    - input_tensor: 输入张量，形状为[B, C, H, W]
    - k: 范围大小，表示周围KxK范围内的点

    返回：
    - 输出张量，形状为[B, KxK-1, H, W]
    """
    B, C, H, W = input_tensor.shape
    # 使用零填充来处理边界情况
    # padded_input = F.pad(input_tensor, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

    # 展平输入张量中每个点及其周围KxK范围内的点
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
    # print(unfold_tensor.shape)
    unfold_tensor = unfold_tensor.reshape(B, C, k**2, H, W)
    # print(unfold_tensor.shape)
    # print(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1].shape,"unfold",k * k//2)
    # 计算余弦相似度
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    # 移除中心点的余弦相似度，得到[KxK-1]的结果
    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

    # 将结果重塑回[B, KxK-1, H, W]的形状
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity


if __name__ == '__main__':
    # x = torch.rand(4, 128, 16, 16)
    # mask = torch.rand(4, 4 * 25, 16, 16)
    # carafe(x, mask, kernel_size=5, group=1, up=2)
    input = torch.randn(1, 128, 512, 512)
    hr_feat = torch.rand(1, 128, 160, 160)
    lr_feat = torch.rand(1, 512, 80, 80)
    model = FreqFusion(hr_channels=128, lr_channels=512)
    params = count_parameters(model)
    print(f"Model parameters: {params}")
    # macs, params_thop = count_flops(model, input)
    # print(f"Model FLOPs: {macs}")
    lr_feat = model(hr_feat=hr_feat, lr_feat=lr_feat)
    # print(mask_lr.shape,hr_feat.shape,lr_feat.shape)
from .Feature_Fusion import *
from ..core import register
# from Feature_Fusion import *

@register()
class Fusion(nn.Module):
    def __init__(self,
                hr_channels_list,
                lr_channels_list,
                scale_factor=1,
                lowpass_kernel=5,
                highpass_kernel=3,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,        
                align_corners=False,
                upsample_mode='nearest',
                feature_resample=True, # use offset generator or not
                feature_resample_group=4,
                comp_feat_upsample=True, # use ALPF & AHPF for init upsampling
                use_high_pass=True,
                use_low_pass=True,
                hr_residual=True,
                semi_conv=True,
                hamming_window=True, # for regularization, do not matter really
                feature_resample_norm=False,
                **kwargs):
        super().__init__()
        print(hr_channels_list,lr_channels_list)
        self.FreqFusion_1 = FreqFusion(hr_channels_list[0],lr_channels_list[0], compressed_channels=compressed_channels//4,semi_conv=semi_conv)
        
        self.FreqFusion_2 = FreqFusion(hr_channels_list[1],lr_channels_list[1],compressed_channels=compressed_channels//2,semi_conv=semi_conv) 
        
        # self.FreqFusion_3 = FreqFusion(hr_channels_list[2],lr_channels_list[2],compressed_channels=compressed_channels,semi_conv=semi_conv)    
    def forward(self, hr_feat_list, lr_feat_list): # use check_point to save GPU memory
        Feature_encoder=[]
        Feature_encoder.append(self.FreqFusion_1(hr_feat_list[0] ,lr_feat_list[1]))
        Feature_encoder.append(self.FreqFusion_2(hr_feat_list[1] ,lr_feat_list[2]))
        # Feature_encoder.append(self.FreqFusion_3(hr_feat_list[3] ,lr_feat_list[3]))
        

        return Feature_encoder

if __name__ == '__main__':

    hr_feat_1 = torch.rand(1, 128, 80, 80)
    lr_feat_1 = torch.rand(1, 512, 80, 80)

    
    hr_feat_2 = torch.rand(1, 256, 40, 40)
    lr_feat_2 = torch.rand(1, 1024, 40, 40)

    # hr_feat_3 = torch.rand(1, 512, 40, 40)
    # lr_feat_3 = torch.rand(1, 2048, 20, 20)
    hr_feat_list=[hr_feat_1,hr_feat_2]
    lr_feat_list=[lr_feat_1,lr_feat_2]
    hr_channels_list=[128,256]
    lr_channels_list=[512,1024]
    # print(hr_channels_list)
    model = Fusion(hr_channels_list, lr_channels_list)
    # params = count_parameters(model)
    # print(f"Model parameters: {params}")
    # macs, params_thop = count_flops(model, input)
    # print(f"Model FLOPs: {macs}")
    Feature_encoder = model(hr_feat_list, lr_feat_list)
    print(Feature_encoder[0].shape,Feature_encoder[1].shape)
    # print(mask_lr.shape,hr_feat.shape,lr_feat.shape)
        
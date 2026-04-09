""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from typing import Any, Dict, List, Optional

from ._transforms import EmptyTransform
from ...core import register, GLOBAL_CONFIG
from skimage.io import imsave

@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], GLOBAL_CONFIG[name]['_name'])(**op)
                    transforms.append(transfom)
                    op['type'] = name

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]
 
        super().__init__(transforms=transforms)

        if policy is None:
            policy = {'name': 'default'}

        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy['name'])(*inputs)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']

        # image=sample[0]
        # image.save('/root/autodl-tmp/Super_re/contain1.jpg', 'JPEG')


        for transform in self.transforms:
            # print(transform)
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        ####在变换之前和变换之后造成了保存图像的问题，仔细排查
        # outputs_save=(sample[0]*255).cpu().detach().numpy().astype('uint8')
        # outputs_save=outputs_save[:, :, :].squeeze()
        # outputs_save=outputs_save.transpose(1,2,0)
        # imsave('/root/autodl-tmp/Super_re/' +'contain2.jpg',outputs_save)

        return sample


    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample

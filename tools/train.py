"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # print(args.config)
    #/root/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_dsp_3x_coco.yml
    update_dict = yaml_utils.parse_cli(args.update)

    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})
    # print('update_dict',update_dict)
    ##本文件添加的配置覆盖源文件中配置
    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)###CFG文件中有所有配置文件的配置，包括yaml里面用include 关键字包括进来的配置文件

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # priority 0
    parser.add_argument('--config',type=str,default='/root/rtdetrv2_pytorch_Super_2/configs/rtdetrv2/rtdetrv2_r50vd_super.yml')
    # parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-r', '--resume', type=str,default='/root/autodl-tmp/OUTPUT2/checkpoint0066.pth', help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str,help='tuning from checkpoint')
    parser.add_argument('--d', type=str , default='[0]' ,help='device')
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)

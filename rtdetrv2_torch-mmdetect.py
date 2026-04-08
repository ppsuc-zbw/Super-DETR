"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T
import os
import numpy as np 
from PIL import Image, ImageDraw
import cv2
from src.core import YAMLConfig
import numpy as np
import random
from mmengine.visualization import Visualizer

def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) // (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales
classes=(
                'airplane',
                'bridge',
                'storage-tank',
                'ship',
                'swimming-pool',
                'vehicle',
                'person',
                'wind-mill',
            )
palette=[
                (
                    56,
                    56,
                    255,
                ),
                (
                    151,
                    157,
                    255,
                ),
                (
                    31,
                    112,
                    255,
                ),
                (
                    29,
                    178,
                    255,
                ),
                (
                    49,
                    210,
                    207,
                ),
                (
                    10,
                    249,
                    72,
                ),
                (
                    23,
                    204,
                    146,
                ),
                (
                    134,
                    219,
                    61,
                ),
            ]


def plot_one_box(xyxy, im0, label=None, color=None, line_thickness=None,blk=np.zeros(3,np.uint8)):

    im0 = cv2.cvtColor(np.asarray(im0),cv2.COLOR_RGB2BGR)
    
    vis.set_image(im0)
    vis.draw_bboxes(xyxy.cpu().detach().numpy())

    # color = color 
    # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

    # areas = (xyxy[3].cpu().detach().numpy() - xyxy[1].cpu().detach().numpy()) * (
    #             xyxy[2].cpu().detach().numpy() - xyxy[0].cpu().detach().numpy())
    # scales = _get_adaptive_scales(areas)
   
    # if label:
    #     # tf = max(tl - 1, 1)  # font thickness
    #     tf= tl
    #     # t_size= int(13 * scales)
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl/3 , thickness=1)[0] 
    #     c2 = c1[0] + t_size[0], c1[1] + t_size[1] 
    #     # c2 = c1[0] , c1[1]  
    #      ## 放入文本还是不展示文本 c1 c2是位置坐标
    #     # cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled color 是文本框背景颜色
        
    #     black=(32,32,32)
    #     cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #     # im0 = cv2.addWeighted(im0, 1.0, blk, 0.7, 1)
    #     ##cv2.puttext坐标表示的是输入文本左下角的框，因此此处要加掉t_size
    #     cv2.putText(im0, label, (c1[0], c2[1]), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
    return vis.get_image()

def draw(vis,images, labels, boxes, scores, thrh = 0.45):

        scores = scores.squeeze()
        labels = labels.squeeze()[scores > 0.45]
        boxes  = boxes.squeeze()[scores > 0.45]
        



   
        im0 = np.asarray(images)
        # im0 = cv2.cvtColor(np.asarray(images),cv2.COLOR_RGB2BGR)

        vis.set_image(im0)
        max_label = int(max(labels) if len(labels) > 0 else 0)

        text_colors = [palette[label] for label in labels]

      
        colors = [palette[label] for label in labels]
            # print("bboxes",bboxes)##传入四个值的张量
        vis.draw_bboxes(
                boxes,
                edge_colors=colors,
                alpha=0.8,
                line_widths=3)

        positions = boxes[:, :2] + 3
        areas = (boxes[:, 3].cpu().detach().numpy() - boxes[:, 1].cpu().detach().numpy()) * (
                boxes[:, 2].cpu().detach().numpy() - boxes[:, 0].cpu().detach().numpy())
        scales = _get_adaptive_scales(areas)
            
        for i, (pos, label) in enumerate(zip(positions, labels)):

          

            label_text = classes[
                        label] if classes is not None else f'class {label}'

            score = round(float(scores[i]) * 100, 1)
            label_text += f': {score}'
                # print(colors[i])
                # self.draw_texts(
                #     label_text,
                #     pos,
                #     colors=text_colors[i],
                #     font_sizes=int(13 * scales[i]),
                #     bboxes=[{
                #         'facecolor': 'black',
                #         'alpha': 0.8,
                #         'pad': 0.7,
                #         'edgecolor': 'none'
                #     }])
                # print(pos)  
            vis.draw_texts(
                    label_text,
                    pos,
                    colors=(255,255,255),
                    font_sizes=int(13 * scales[i]),
                    bboxes=[{
                        'facecolor': tuple(value / 255 for value in colors[i])+(0,),
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])
        im=vis.get_image()
        return im
        
            

def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs,x1 = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs,x1

    model = Model().to(args.device)
    imagelist=os.listdir(args.im_file)
    vis = Visualizer()
    
    for item in imagelist:
        path='/root/autodl-tmp/Test_result/'+item
        path_s='/root/autodl-tmp/Super_result/'+item
        im_pil = Image.open(args.im_file+'/'+item)
        # im_pil = Image.open(args.im_file+'/'+item).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)

        output,x1 = model(im_data, orig_size)


        
        labels, boxes, scores = output
        im = draw(vis,im_pil, labels, boxes, scores, thrh = 0.45)
        im = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        # cv2.imwrite(path,im) # detection
        print(x1.max())

        # x1=x1/(x1.max()-x1.min())
        outputs_save=(x1*255).cpu().detach().numpy().astype('uint8')
        # outputs_save=outputs_save
        outputs_save = np.clip(outputs_save, 0, 255)
        # print(outputs_save[0:1,:, :, :].squeeze().shape)
        outputs_save=outputs_save[0:1,:, :, :].squeeze()
        outputs_save=outputs_save.transpose(1,2,0)
        # print(outputs_save.max(),outputs_save.min())
        cv2.cvtColor(np.asarray(outputs_save),cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_s,outputs_save)
        # Image.save(path+'.png', im)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,default='/root/rtdetrv2_pytorch_Super_2/configs/rtdetrv2/rtdetrv2_r50vd_super.yml' )
    parser.add_argument('-r', '--resume', type=str,default='/root/autodl-tmp/OUTPUT2/checkpoint0000.pth' )
    parser.add_argument('-f', '--im-file', type=str,default='/root/autodl-tmp/aitod/images/test/' )
    # parser.add_argument('-f', '--im-file', type=str,default='/root/rtdetrv2_pytorch_Super_2/2' )
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)

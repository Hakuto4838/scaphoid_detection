import torch
import torch.nn as nn
import torchvision
import math

from tools.tool import *
from tools.RPN import RPN
from tools.ROI import ROIHead

class FasterRCNN(nn.Module):
    def __init__(self, num_classes = 2):
        super(FasterRCNN, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained = True)

        self.backbone = vgg16.features[:-1]
        self.rpn = RPN()
        self.roi_head = ROIHead(num_classes=num_classes)

        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        self.img_mean = [0.5]
        self.img_std =  [0.3]
        self.min_size = 1200
        self.max_size = 1400

    def norm_resize_image(self, image, bboxes):
        mean = torch.as_tensor(self.img_mean, dtype=image.dtype, device=image.device)
        std  = torch.as_tensor(self.img_std , dtype=image.dtype, device=image.device)
        image = (image-mean[:, None, None] ) / std[:, None, None]

        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(
            float(self.min_size) / min_size,
            float(self.max_size) / max_size
        ).item()

        image = torch.nn.functional.interpolate(
            image,
            size = None,
            scale_factor= scale,
            mode='bilinear',
            recompute_scale_factor=True,
            align_corners=False
        )

        if bboxes is not None:
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h,w)) 
            ]
            ratio_h, ratio_w = ratios

            x_min, y_min, x_max, y_max = bboxes.unbind(2)
            x_min = x_min * ratio_w
            y_min = y_min * ratio_h
            x_max = x_max * ratio_w
            y_max = y_max * ratio_h
            bboxes = torch.stack((
                x_min, y_min, x_max, y_max
            ), dim=2)
        return image, bboxes


    def forward(self, image, target=None):
        if not isinstance(image, torch.Tensor):
            image = torchvision.transforms.ToTensor()(image)
            device = next(self.backbone.parameters()).device
            image = image.to(device=device)
            if image.ndim == 3:
                image = image.unsqueeze(0)

        old_shape = image.shape[-2:]
        if not target is None:
            image, bboxes = self.norm_resize_image(
                image, target['bboxes']
            )
            target['bboxes'] = bboxes
        else :
            image, _ = self.norm_resize_image(
                image, None
            )

        image_3ch = image.repeat(1, 3, 1, 1)
        feat = self.backbone(image_3ch)

        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']

        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)

        if not self.training:
            frcnn_output['boxes'] = tf_boxes_2_original_size(
                frcnn_output['boxes'],
                image.shape[-2:],
                old_shape
            )

        return rpn_output, frcnn_output
    

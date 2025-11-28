import torch
import torch.nn as nn
import torchvision
import math

from tools.tool import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RPN(nn.Module):
    def __init__(self, in_channels = 512):
        super(RPN, self).__init__()

        self.scales = [64, 128, 256]
        self.aspect_ratios = [0.5, 1.0, 1.5]
        self.num_anchors = 9

        self.rpn_conv = nn.Conv2d(in_channels, in_channels, 
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.cls_layer = nn.Conv2d(in_channels,
                                   self.num_anchors,
                                   kernel_size=1,
                                   stride=1)
        self.bbox_reg_layer = nn.Conv2d(in_channels,
                                        self.num_anchors*4,
                                        kernel_size=1,
                                        stride=1)
    
    def gen_anchors(self, image, feat):
        grid_h, grid_w = feat.shape[-2:]
        img_h, img_w = image.shape[-2:]

        strid_h = torch.tensor(img_h//grid_h, dtype=torch.int64, device=feat.device)
        strid_w = torch.tensor(img_w//grid_w, dtype=torch.int64, device=feat.device)

        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)

        #h*w = 1
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1/h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1)/2
        base_anchors = base_anchors.round()

        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * strid_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * strid_h
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)

        shift = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1) #H_feat*W_feat * 4
        anchors = (shift.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        anchors = anchors.reshape(-1, 4)

        return anchors

        
    def forward(self, image, feat, target):
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        anchors = self.gen_anchors(image, feat)

        nAnchor_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0,2,3,1) #通道擺後面
        cls_scores = cls_scores.reshape(-1, 1)

        box_transform_pred= box_transform_pred.view(
            box_transform_pred.size(0),
            nAnchor_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1]
        ) # batch, 每個錨點 anchors 數量, 4, Hfeat, Wfeat
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2).reshape(-1, 4)

        proposals = apply_reg_pred_to_anchor(box_transform_pred.detach().reshape(-1, 1, 4), anchors)

        proposals = proposals.reshape(proposals.size(0), 4)
        proposals, scores = self.filter_proposals(proposals, cls_scores.detach(), image.shape)

        rpn_output = {
            'proposals' : proposals,
            'scores' : scores
        }

        if not self.training or target is None:
            return rpn_output
        else:
            #training
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target['bboxes'][0]
            )

            regression_tgt = boxes_to_transformation_targets(
                matched_gt_boxes_for_anchors, anchors
            )

            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_pn(labels_for_anchors, positive_count=128, total_count=256)

            sample_idxs = torch.where(sampled_pos_idx_mask|sampled_neg_idx_mask)[0]

            local_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[sampled_pos_idx_mask],
                    regression_tgt[sampled_pos_idx_mask],
                    beta= 1/9,
                    reduction='sum'
                ) / sample_idxs.numel()
            )

            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_scores[sample_idxs].flatten(),
                labels_for_anchors[sample_idxs].flatten()
            )

            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = local_loss
            return rpn_output


    def assign_targets_to_anchors(self, anchors, gt_boxes):
        iou_matrix = get_IOU(gt_boxes, anchors)
        best_matrix_iou, best_match_gt_index = iou_matrix.max(dim=0)
        best_match_gt_idx_pre_threshold = best_match_gt_index.clone()

        below_low_threshold = best_matrix_iou < 0.3
        between_threshold = (best_matrix_iou>=0.3) & (best_matrix_iou<0.7)
        best_match_gt_index[below_low_threshold] = -1
        best_match_gt_index[between_threshold] = -2

        #對 gt 而言最大的 anchor 的 iou
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
        gt_pred_pair_with_hightest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])

        pred_inds_to_updata = gt_pred_pair_with_hightest_iou[1]
        best_match_gt_index[pred_inds_to_updata] = best_match_gt_idx_pre_threshold[pred_inds_to_updata]

        best_match_gt_boxes = gt_boxes[best_match_gt_index.clamp(min=0)]

        labels = best_match_gt_index >= 0
        labels = labels.to(dtype=torch.float32)

        background_anchors = best_match_gt_index == -1
        labels[background_anchors] = 0.0

        ignored_anchors = best_match_gt_index == -2
        labels[ignored_anchors] = -1.0

        return labels, best_match_gt_boxes


    def filter_proposals(self, proposals, cls_scores, image_shape):
        #只保留前10000個資料
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(10000)
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]


        proposals = clamp_boundary(proposals, image_shape)

        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]

        #nms
        # keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals , cls_scores, 0.7)

        post_nms_keep_indices = keep_indices[
            cls_scores[keep_indices].sort(descending=True)[1]
        ]

        proposals = proposals[post_nms_keep_indices[:2000]]
        cls_scores = cls_scores[post_nms_keep_indices[:2000]]

        return proposals, cls_scores


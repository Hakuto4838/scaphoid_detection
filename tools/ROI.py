import torch
import torch.nn as nn
import torchvision
import math

from tools.tool import *

class ROIHead(nn.Module):
    def __init__(self, num_classes=2, in_channels = 512):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.pool_size=7
        self.fc_inner_dim=1024

        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)

        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes*4)

    def assign_target_to_proposals(seelf, proposals, gt_boxes, gt_labels):
        iou_matrix = get_IOU(gt_boxes, proposals)
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0) #對於proposals最接近的 gt

        below_low_th = best_match_iou < 0.5 #忽略過低
        best_match_gt_idx[below_low_th] = -1
        match_gt_box_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]

        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)

        background_proposals = best_match_gt_idx == -1
        labels[background_proposals] = 0

        return labels, match_gt_box_for_proposals
    
    def filter_pred(self, pred_boxes, pred_labels, pred_scores):
        #過濾掉分數太低的
        keep = torch.where(pred_scores > 0.05)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        #box size 太小
        min_size = 1
        ws, hs =  pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        # NMS 
        keep = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(
                pred_boxes[curr_indices],
                pred_scores[curr_indices],
                0.5
            )
            keep[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep)[0]
        #排序篩選出來的pred_scores，取其 index，用來排列 keep_indices
        # 反正 keep_indices 會排序成 pred_scores 的遞減排列索引 (還有篩選)
        post_nms_keep_idxs = keep_indices[pred_scores[keep_indices].sort(
            descending=True
        )[1]]

        keep = post_nms_keep_idxs[:100] #取前 100 
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        return pred_boxes, pred_labels, pred_scores        


    def forward(self, feat, proposals, img_shape, target):
        if self.training and target is not None:
            gt_boxes = target['bboxes'][0]
            gt_labels =target['labels'][0]

            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(
                proposals, gt_boxes, gt_labels)
            sample_neg_mask, sample_pos_mask = sample_pn(
                labels, positive_count=32, total_count=128
            )
            sample_idxs = torch.where(sample_pos_mask | sample_neg_mask)


            proposals = proposals[sample_idxs]
            labels = labels[sample_idxs]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sample_idxs]
            regression_tgt = boxes_to_transformation_targets(
                matched_gt_boxes_for_proposals, proposals
            )

        spatial_scale = 1/16 #VGG16

        proposals_roi_pool_feat = torchvision.ops.roi_pool(
            feat,
            [proposals], 
            output_size = self.pool_size,
            spatial_scale = spatial_scale
        )

        proposals_roi_pool_feat = proposals_roi_pool_feat.flatten(start_dim=1)
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposals_roi_pool_feat))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))

        cls_scores = self.cls_layer(box_fc_7)
        boxes_tf_pred = self.bbox_reg_layer(box_fc_7)

        #形狀統一
        num_boxes, num_classes = cls_scores.shape
        boxes_tf_pred = boxes_tf_pred.reshape(num_boxes, num_classes, 4)

        frcnn_output = {}

        if self.training and target is not None:
            cls_loss = torch.nn.functional.cross_entropy(
                cls_scores,
                labels
            )
            fg_proposal_idxs = torch.where(labels > 0)[0] #非背景區塊
            fg_cls_labels = labels[fg_proposal_idxs]
            local_loss = torch.nn.functional.smooth_l1_loss(
                boxes_tf_pred[fg_proposal_idxs, fg_cls_labels],
                regression_tgt[fg_proposal_idxs],
                beta=1/9,
                reduction='sum'
            )
            assert labels.numel()!=0
            local_loss = local_loss / labels.numel()
            assert not torch.isinf(local_loss), f"inf at {local_loss}/ {labels.numel()} "

            frcnn_output['frcnn_cls_loss'] = cls_loss
            frcnn_output['frcnn_loc_loss'] = local_loss

            return frcnn_output
        
        else:
            pred_boxes = apply_reg_pred_to_anchor(
                boxes_tf_pred,
                proposals
            )

            pred_scores = torch.nn.functional.softmax(cls_scores, dim=1)

            pred_boxes =clamp_boundary(pred_boxes, img_shape)
            pred_labels = torch.arange(num_classes, device=cls_scores.device)
            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)

            #去除背景資訊
            pred_boxes = pred_boxes[:, 1:] 
            pred_scores = pred_scores[:, 1:]
            pred_labels = pred_labels[:, 1:]
            # pred_boxes -> (num_proposals, numclass-1, 4)

            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)

            pred_boxes, pred_labels, pred_scores = self.filter_pred(pred_boxes, pred_labels, pred_scores)

            frcnn_output['boxes'] = pred_boxes 
            frcnn_output['scores'] = pred_scores
            frcnn_output['labels'] = pred_labels

            return frcnn_output



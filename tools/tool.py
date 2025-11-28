import torch
import torch.nn as nn
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

def apply_reg_pred_to_anchor(box_transform_pred, anchors_or_proposals):
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)

    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]

    center_x = anchors_or_proposals[:, 0] + 0.5*w
    center_y = anchors_or_proposals[:, 1] + 0.5*h 

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]

    pred_center_x = dx*w[:, None] + center_x[:, None]
    pred_center_y = dy*h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:,None]
    pred_h = torch.exp(dh) * h[:,None]

    pres_boxes = torch.stack((
        pred_center_x - 0.5 * pred_w,
        pred_center_y - 0.5 * pred_h,
        pred_center_x + 0.5 * pred_w,
        pred_center_y + 0.5 * pred_h
    ),dim=2)

    return pres_boxes

def clamp_boundary(boxes, image_shape):
    #處理超過邊界問題
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]

    boxes_x1 = boxes_x1.clamp(min=0, max=image_shape[-1])
    boxes_y1 = boxes_y1.clamp(min=0, max=image_shape[-2])
    boxes_x2 = boxes_x2.clamp(min=0, max=image_shape[-1])
    boxes_y2 = boxes_y2.clamp(min=0, max=image_shape[-2])

    return torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None],
    ), dim=-1)

def get_IOU(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    x_left = torch.max(box1[:, None, 0], box2[:, 0])
    y_top = torch.max(box1[:, None, 1], box2[:, 1])
    x_right = torch.min(box1[:, None, 2], box2[:, 2])
    y_bottom = torch.min(box1[:, None, 3], box2[:, 3])

    intersection_area = (x_right-x_left).clamp(min=0) * (y_bottom-y_top).clamp(min=0)
    union = area1[:,None] + area2 - intersection_area
    return intersection_area / union

def boxes_to_transformation_targets(gt_boxes, anchors):
    #計算 anchors 到 gt 的轉換距離
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    center_x = anchors[:, 0] + 0.5 * widths
    center_y = anchors[:, 1] + 0.5 * heights

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_center_y = gt_boxes[:, 1] + 0.5 * gt_h

    target_dx = (gt_center_x - center_x) / widths
    target_dy = (gt_center_y - center_y) / heights
    target_dw = torch.log(gt_w / widths)
    target_dh = torch.log(gt_h / heights)

    result =  torch.stack((
        target_dx,
        target_dy,
        target_dw,
        target_dh
    ), dim=1)

    assert not torch.isinf(result).any(), f"inf in {result}"

    return result


def sample_pn(labels, positive_count, total_count):
    #採樣正負標籤樣本
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0] #背景
    num_pos = min(positive.numel(), positive_count)
    num_neg = min(negative.numel(), total_count-num_pos)

    perm_pos_idx = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_neg_idx = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idxs = positive[perm_pos_idx]
    neg_idxs = negative[perm_neg_idx]

    sample_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sample_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sample_pos_idx_mask[pos_idxs] = True
    sample_neg_idx_mask[neg_idxs] = True
    return sample_neg_idx_mask, sample_pos_idx_mask

def tf_boxes_2_original_size(boxes, new_size, original_size):
    ratios_h, ratios_w = [
                torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) /
                torch.tensor(s, dtype=torch.float32, device=boxes.device)
                for s, s_orig in zip(new_size, original_size) 
            ]
    
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_w
    ymin = ymin * ratios_h
    xmax = xmax * ratios_w
    ymax = ymax * ratios_h

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def points_to_rotated_box(points):
    # Convert to numpy array for easier manipulation
    points = np.array(points)
    
    center = np.mean(points, axis=0)
    cx, cy = center
    
    width_1 = np.linalg.norm(points[1] - points[0])  # Top edge
    width_2 = np.linalg.norm(points[2] - points[3])  # Bottom edge
    width = (width_1 + width_2) / 2

    height_1 = np.linalg.norm(points[3] - points[0])  # Left edge
    height_2 = np.linalg.norm(points[2] - points[1])  # Right edge
    height = (height_1 + height_2) / 2
    
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    angle_rad = math.atan2(dy, dx)
    
    return (float(cx), float(cy), float(width), float(height), float(angle_rad))

def rotated_box_to_points(cx, cy, width, height, angle_rad):
    """
    Convert rotated box parameters to 4 corner points.
    Args:
        cx, cy: Center coordinates
        width, height: Box dimensions
        angle_rad: Rotation angle in rad
    Returns:
        List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    
    # Calculate the four corners relative to center (0,0)
    half_w = width / 2
    half_h = height / 2
    corners_rel = np.array([
        [-half_w, -half_h],  # top-left
        [half_w, -half_h],   # top-right
        [half_w, half_h],    # bottom-right
        [-half_w, half_h]    # bottom-left
    ])
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)]
    ])
    
    # Rotate corners and add center offset
    rotated_corners = np.dot(corners_rel, rotation_matrix.T)
    points = rotated_corners + np.array([cx, cy])
    
    return points.tolist()

def draw_rotated_box(im, bbox=None, idx=0):
    fig, ax = plt.subplots(1)
    ax.imshow(im.permute(1, 2, 0).numpy(), cmap='gray')  # 转换 Tensor 为 numpy 格式显示

    if not bbox is None:
        cx, cy, width, height, angle_rad = bbox
        angle_deg = angle_rad * 180 / np.pi
        # 創建矩形
        rect = Rectangle(
            (-width/2, -height/2),  # 從中心點開始
            width, height,
            facecolor='none',
            edgecolor='r',
            linewidth=2
        )
        
        # 創建變換
        t = Affine2D() \
            .rotate_deg(angle_deg) \
            .translate(cx, cy)
        
        # 應用變換
        rect.set_transform(t + ax.transData)
        
        ax.add_patch(rect)
        # rect = Rectangle(
        #     (cx - width / 2, cy - height / 2), width, height,
        #     angle=angle_deg, edgecolor='r', facecolor='none', linewidth=2
        # )

        # ax.add_patch(rect)
    # plt.show()
    plt.axis('off')
    plt.savefig(f"testimg/{idx}", bbox_inches='tight', pad_inches=0)
    plt.close()

def draw_box_from_points(im, points=None, idx=0):
    """
    繪製旋轉矩形，使用4個頂點座標
    """
    fig, ax = plt.subplots(1)
    img_np = im.permute(1, 2, 0).numpy()
    ax.imshow(img_np, cmap='gray')

    if points is not None:
        # 將巢狀列表中的 tensor 轉換為 numpy array
        corners = np.array([[float(point[0].cpu().numpy()), float(point[1].cpu().numpy())] 
                           for point in points])
        
        # 使用 matplotlib.patches.Polygon 來繪製四邊形
        from matplotlib.patches import Polygon
        poly = Polygon(corners,
                      fill=False,
                      edgecolor='r',
                      linewidth=2)
        
        ax.add_patch(poly)
        
        # 設定適當的顯示範圍
        ax.set_xlim(0, img_np.shape[1])
        ax.set_ylim(img_np.shape[0], 0)
        

    plt.axis('off')
    plt.savefig(f"testimg/{idx}", bbox_inches='tight', pad_inches=0)
    plt.close()


def apply_reg_pred_to_anchor_r(box_transform_pred, anchors_or_proposals):
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 5)

    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]

    center_x = anchors_or_proposals[:, 0] + 0.5*w
    center_y = anchors_or_proposals[:, 1] + 0.5*h 
    original_angle = anchors_or_proposals[:, 4]

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]
    da = box_transform_pred[..., 4]  # angle offset

    pred_center_x = dx*w[:, None] + center_x[:, None]
    pred_center_y = dy*h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:,None]
    pred_h = torch.exp(dh) * h[:,None]

    pred_angle = da + original_angle[:, None]

    pred_angle = torch.fmod(pred_angle, 2 * torch.pi)

    pred_x1 = pred_center_x - 0.5 * pred_w
    pred_y1 = pred_center_y - 0.5 * pred_h
    pred_x2 = pred_center_x + 0.5 * pred_w
    pred_y2 = pred_center_y + 0.5 * pred_h

    # Stack all parameters including angle
    pred_boxes = torch.stack((
        pred_x1,
        pred_y1,
        pred_x2,
        pred_y2,
        pred_angle
    ), dim=2)

    return pred_boxes


def get_IOU_r(box1, box2):
    """
    Calculate IoU between rotated bounding boxes
    box1, box2: [N,5] and [M,5], (x1,y1,x2,y2,angle)
    return: IoU [N,M]
    """
    def get_rotated_corners(boxes):
        # Convert from x1,y1,x2,y2,angle to cx,cy,w,h,angle
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        angle = boxes[:, 4]
        
        # Get corners (counter-clockwise order)
        corners = torch.zeros((boxes.shape[0], 4, 2), device=boxes.device)
        
        # Calculate the cos and sin of angle
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Calculate the four corners
        # top-left, top-right, bottom-right, bottom-left
        corners[:, 0, 0] = cx - w/2  # x coordinates
        corners[:, 0, 1] = cy - h/2  # y coordinates
        corners[:, 1, 0] = cx + w/2
        corners[:, 1, 1] = cy - h/2
        corners[:, 2, 0] = cx + w/2
        corners[:, 2, 1] = cy + h/2
        corners[:, 3, 0] = cx - w/2
        corners[:, 3, 1] = cy + h/2
        
        # Rotate the corners
        for i in range(4):
            x = corners[:, i, 0] - cx
            y = corners[:, i, 1] - cy
            corners[:, i, 0] = cx + (x * cos_a - y * sin_a)
            corners[:, i, 1] = cy + (x * sin_a + y * cos_a)
        
        return corners

    def polygon_area(corners):
        # Calculate area using the Shoelace formula
        n = corners.shape[1]  # number of corners
        area = corners[:, 0, 0] * corners[:, 1, 1] - corners[:, 1, 0] * corners[:, 0, 1]
        for i in range(1, n-1):
            area += corners[:, i, 0] * corners[:, i+1, 1] - corners[:, i+1, 0] * corners[:, i, 1]
        area += corners[:, -1, 0] * corners[:, 0, 1] - corners[:, 0, 0] * corners[:, -1, 1]
        return torch.abs(area) / 2

    def compute_intersection_area(corners1, corners2):
        # Use Sutherland-Hodgman algorithm for polygon clipping
        def clip_polygon(subject_polygon, clip_polygon):
            def inside(p, cp1, cp2):
                return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

            def compute_intersection(p1, p2, cp1, cp2):
                dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
                dp = [p1[0] - p2[0], p1[1] - p2[1]]
                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                n2 = p1[0] * p2[1] - p1[1] * p2[0]
                n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
                return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

            output_list = subject_polygon
            cp1 = clip_polygon[-1]

            for j in range(len(clip_polygon)):
                cp2 = clip_polygon[j]
                input_list = output_list
                output_list = []
                if not input_list:
                    break
                s = input_list[-1]

                for i in range(len(input_list)):
                    e = input_list[i]
                    if inside(e, cp1, cp2):
                        if not inside(s, cp1, cp2):
                            output_list.append(compute_intersection(s, e, cp1, cp2))
                        output_list.append(e)
                    elif inside(s, cp1, cp2):
                        output_list.append(compute_intersection(s, e, cp1, cp2))
                    s = e
                cp1 = cp2
            return output_list

        # Convert corners to list of points
        poly1 = corners1.tolist()
        poly2 = corners2.tolist()
        
        # Compute intersection polygon
        intersection_poly = clip_polygon(poly1, poly2)
        
        if not intersection_poly:
            return 0.0
            
        # Convert intersection points back to tensor and compute area
        intersection_corners = torch.tensor(intersection_poly, device=corners1.device)
        return polygon_area(intersection_corners.unsqueeze(0))[0]

    # Get corners for all boxes
    corners1 = get_rotated_corners(box1)  # [N,4,2]
    corners2 = get_rotated_corners(box2)  # [M,4,2]
    
    # Calculate areas
    area1 = polygon_area(corners1)  # [N]
    area2 = polygon_area(corners2)  # [M]
    
    # Initialize IoU matrix
    N, M = box1.size(0), box2.size(0)
    ious = torch.zeros((N, M), device=box1.device)
    
    # Calculate IoU for each pair of boxes
    for i in range(N):
        for j in range(M):
            intersection = compute_intersection_area(corners1[i], corners2[j])
            union = area1[i] + area2[j] - intersection
            ious[i,j] = intersection / union if union > 0 else 0.0
    
    return ious


def boxes_to_transformation_targets_r(gt_boxes, anchors):
    """
    Calculate transformation targets from anchors to ground truth boxes,
    including rotated angle with periodic handling
    Args:
        gt_boxes: [N, 5] (x1, y1, x2, y2, angle)
        anchors: [N, 5] (x1, y1, x2, y2, angle)
    Returns:
        [N, 5] transformation targets (dx, dy, dw, dh, da)
    """
    # 計算寬高和中心點（與原本相同）
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    center_x = anchors[:, 0] + 0.5 * widths
    center_y = anchors[:, 1] + 0.5 * heights

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_center_y = gt_boxes[:, 1] + 0.5 * gt_h

    # 計算位置和尺寸的變換（與原本相同）
    target_dx = (gt_center_x - center_x) / widths
    target_dy = (gt_center_y - center_y) / heights
    target_dw = torch.log(gt_w / widths)
    target_dh = torch.log(gt_h / heights)

    # 處理角度變換
    anchor_angles = anchors[:, 4]
    gt_angles = gt_boxes[:, 4]
    
    # 將角度差異限制在 [-π, π] 範圍內
    diff_angle = gt_angles - anchor_angles
    diff_angle = torch.atan2(torch.sin(diff_angle), torch.cos(diff_angle))

    # 堆疊所有變換目標
    result = torch.stack((
        target_dx,
        target_dy,
        target_dw,
        target_dh,
        diff_angle
    ), dim=1)

    # 檢查無效值
    assert not torch.isinf(result).any(), f"inf in {result}"
    assert not torch.isnan(result).any(), f"nan in {result}"

    return result

def inverse_boxes_transformation_r(boxes, deltas):
    """
    Apply transformation deltas to boxes
    Args:
        boxes: [N, 5] source boxes
        deltas: [N, 5] transformation deltas
    Returns:
        [N, 5] transformed boxes
    """
    # 提取寬高和中心點
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    center_x = boxes[:, 0] + 0.5 * widths
    center_y = boxes[:, 1] + 0.5 * heights
    
    # 應用變換
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    da = deltas[:, 4]

    # 計算新的中心點和寬高
    pred_center_x = dx * widths + center_x
    pred_center_y = dy * heights + center_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    
    # 計算新的角度
    pred_angle = boxes[:, 4] + da
    
    # 將角度規範化到指定範圍
    pred_angle = torch.atan2(torch.sin(pred_angle), torch.cos(pred_angle))
    
    # 轉換回框的格式
    pred_boxes = torch.stack([
        pred_center_x - 0.5 * pred_w,
        pred_center_y - 0.5 * pred_h,
        pred_center_x + 0.5 * pred_w,
        pred_center_y + 0.5 * pred_h,
        pred_angle
    ], dim=1)
    
    return pred_boxes

def clamp_boundary_r(boxes, image_shape):
    #處理超過邊界問題
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    boxes_rad = boxes[..., 4]

    boxes_x1 = boxes_x1.clamp(min=0, max=image_shape[-1])
    boxes_y1 = boxes_y1.clamp(min=0, max=image_shape[-2])
    boxes_x2 = boxes_x2.clamp(min=0, max=image_shape[-1])
    boxes_y2 = boxes_y2.clamp(min=0, max=image_shape[-2])

    return torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None],
        boxes_rad[..., None]
    ), dim=-1)

def get_IOU_score(gtbox, pdbox):
    areagt = (gtbox[2] - gtbox[0]) * (gtbox[3] - gtbox[1])
    areapd = (pdbox[2] - pdbox[0]) * (pdbox[3] - pdbox[1])

    x_left =   torch.max(gtbox[0], pdbox[0])
    y_top =    torch.max(gtbox[1], pdbox[1])
    x_right =  torch.min(gtbox[2], pdbox[2])
    y_bottom = torch.min(gtbox[3], pdbox[3])

    tt_left =   torch.min(gtbox[0], pdbox[0])
    tt_top =    torch.min(gtbox[1], pdbox[1])
    tt_right =  torch.max(gtbox[2], pdbox[2])
    tt_bottom = torch.max(gtbox[3], pdbox[3])

    intersection_area = (x_right-x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    total_area = (tt_right-tt_left) * (tt_bottom-tt_top)

    nTP = intersection_area
    nFP = areapd - intersection_area
    nFN = areagt - intersection_area
    nTN = total_area - nTP - nFP - nFN

    union = areagt + areapd - intersection_area
    acc = (nTP+nFN) / (nTP + nFP + nFN + nTN)
    prec = nTP / (nTP + nFN)
    recall = nTP / (nTP + nFN)
    #return iou, acc, prec, recall
    return intersection_area / union, acc, prec, recall
import os
import json
import cv2
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def points_to_rotated_box(points):
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

def create_coco_annotation(image_id, bbox_info, category_id=1):
    """將旋轉框資訊轉換為 COCO 格式的標註"""
    cx, cy, w, h, angle = bbox_info
    area = w * h
    
    return {
        'id': image_id,
        'image_id': image_id,
        'category_id': category_id,
        'bbox': [cx, cy, w, h, angle],
        'area': area,
        'iscrowd': 0
    }

def process_dataset(src_dir, output_dir):
    """處理資料集並轉換為COCO格式"""
    print(f"Processing directory: {src_dir}")
    
    # 建立輸出目錄
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # 初始化COCO格式資料結構
    coco_format = {
        'info': {
            'year': 2024,
            'version': '1.0',
            'description': 'Fracture Detection Dataset',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'images': [],
        'annotations': [],
        'categories': [{
            'id': 1,
            'name': 'fracture',
            'supercategory': 'none'
        }]
    }
    
    image_id = 1
    
    # 取得所有圖片檔案
    images_dir = Path(src_dir) / 'images'
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
    image_files = list(images_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} images")
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # 取得相關檔案路徑
            base_name = img_path.stem
            scaphoid_json = os.path.join(src_dir, 'scaphoid', f'{base_name}.json')
            annotation_json = os.path.join(src_dir, 'annotations', f'{base_name}.json')
            
            print(f"Processing image: {base_name}")
            
            # 檢查必要檔案是否存在
            if not os.path.exists(scaphoid_json):
                print(f"Scaphoid JSON not found: {scaphoid_json}")
                continue
            
            # 讀取scaphoid區域資訊
            with open(scaphoid_json, 'r', encoding='utf-8') as f:
                scaphoid_data = json.load(f)
            
            if not scaphoid_data:
                print(f"Empty scaphoid data for: {base_name}")
                continue
                
            scaphoid_bbox = [int(float(x)) for x in scaphoid_data[0]['bbox']]
            
            # 讀取圖片
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            # 確保邊界框座標在有效範圍內
            x1, y1, x2, y2 = scaphoid_bbox
            height, width = img.shape
            x1 = max(0, min(x1, width-1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height-1))
            y2 = max(0, min(y2, height))
            
            if x1 >= x2 or y1 >= y2:
                print(f"Invalid bbox coordinates after adjustment: {[x1, y1, x2, y2]}")
                continue
            
            # 裁剪圖片
            cropped_img = img[y1:y2, x1:x2]
            
            if cropped_img.size == 0:
                print(f"Empty cropped image for: {base_name}")
                continue
            
            # 儲存裁剪後的圖片，保持原始檔名
            output_img_path = os.path.join(output_dir, 'images', f'{base_name}.jpg')
            success = cv2.imwrite(output_img_path, cropped_img)
            if not success:
                print(f"Failed to save image: {output_img_path}")
                continue
            
            # 添加圖片資訊
            coco_format['images'].append({
                'id': image_id,
                'file_name': f'{base_name}.jpg',  # 使用原始檔名
                'width': cropped_img.shape[1],
                'height': cropped_img.shape[0],
            })
            
            # 處理標註資訊
            if os.path.exists(annotation_json):
                with open(annotation_json, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                
                if annotation_data and annotation_data[0]['name'] == 'Fracture':
                    bbox_points = annotation_data[0]['bbox']
                    rotated_bbox = points_to_rotated_box(bbox_points)
                    annotation = create_coco_annotation(image_id, rotated_bbox)
                    coco_format['annotations'].append(annotation)
            
            image_id += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    # 儲存COCO格式的標註檔
    annotation_path = os.path.join(output_dir, 'annotations.json')
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {image_id-1} images")
    print(f"Created {len(coco_format['annotations'])} annotations")
    print(f"Saved annotations to {annotation_path}")

def main():
    try:
        # 處理訓練集和測試集
        process_dataset(
            src_dir='fracture_detection/train',
            output_dir='processed_data/train'
        )
        process_dataset(
            src_dir='fracture_detection/test',
            output_dir='processed_data/val'
        )
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == '__main__':
    main()
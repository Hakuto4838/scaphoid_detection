import glob
import os
import random
import json

import torch
import torchvision
from PIL import Image, ImageEnhance
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
""" use
test_dataset = ScaphoidDataset(
    split='test',
    im_dir='test/images',
    ann_dir='test/annotations'
)

dataset = ScaphoidDataset(
    split='train',
    im_dir='train/images',
    ann_dir='train/annotations',
    contrast_range=(0.5, 1.5)
)
"""


def load_images_and_anns(im_dir, ann_dir):
    """
    載入圖片和標註資訊
    :param im_dir: 圖片目錄路徑
    :param ann_dir: 標註檔目錄路徑
    :return: 資料清單
    img_id : name
    filename : path
    width, height : size
    detections : list of det
        det : 
            
    """
    im_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.json'))):
        im_info = {}
        base_name = os.path.basename(ann_file).split('.json')[0]
        im_info['img_id'] = base_name
        im_info['filename'] = os.path.join(im_dir, f'{base_name}.jpg')
        
        if not os.path.exists(im_info['filename']):
            print(f"Warning: image {im_info['filename']} not found")
            continue
            
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)
            
        im = Image.open(im_info['filename'])
        im_info['width'], im_info['height'] = im.size
        
        detections = []
        for obj in ann_data:
            det = {}
            det['label'] = 1  # Scaphoid class
            bbox = [
                int(float(obj['bbox'][0])),
                int(float(obj['bbox'][1])),
                int(float(obj['bbox'][2])),
                int(float(obj['bbox'][3]))
            ]
            det['bbox'] = bbox
            detections.append(det)
            
        im_info['detections'] = detections
        im_infos.append(im_info)
        
    print(f'Total {len(im_infos)} images found')
    return im_infos

class ScaphoidDataset(Dataset):
    def __init__(self, split, im_dir, ann_dir, contrast_range=(0.5, 1.5)):
        """
        初始化資料集
        :param split: 'train' 或 'test'
        :param im_dir: 圖片目錄路徑
        :param ann_dir: 標註檔目錄路徑
        :param contrast_range: 對比度調整範圍的元組 (最小值, 最大值)
        """
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        self.contrast_range = contrast_range
        
        self.classes = ['background', 'Scaphoid']
        self.label2idx = {self.classes[idx]: idx for idx in range(len(self.classes))}
        self.idx2label = {idx: self.classes[idx] for idx in range(len(self.classes))}
        print("Class mapping:", self.idx2label)
        
        self.images_info = load_images_and_anns(im_dir, ann_dir)
    
    def apply_augmentation(self, image):
        """
        應用資料增強
        :param image: PIL Image
        :return: 增強後的圖片
        """
        # 水平翻轉
        to_flip = False
        if random.random() < 0.5:
            to_flip = True
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
        # 對比度調整
        if random.random() < 0.5:
            contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
            
        return image, to_flip
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        im_info = self.images_info[index]
        fname = im_info['filename']
        # 讀取圖片
        im = Image.open(im_info['filename'])
        
        # 訓練時進行資料增強
        to_flip = False
        if self.split == 'train':
            im, to_flip = self.apply_augmentation(im)
            
        # 轉換為張量
        im_tensor = torchvision.transforms.ToTensor()(im)
        
        # 準備標註資訊
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])
        
        # 如果進行了翻轉，調整bbox座標
        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2-x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])

        for idx, box in enumerate(targets['bboxes']):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # 確保寬高為正
            assert w > 0, f"Invalid box width at {fname}: {box}"
            assert h > 0, f"Invalid box height at {fname}: {box}"
            
            # 確保在圖片範圍內
            assert x1 >= 0 and y1 >= 0, f"Box coordinates negative at {fname}: {box} : {im_tensor.shape[-2:]}"
            assert x2 <= im_tensor.shape[-1] and y2 <= im_tensor.shape[-2], \
                f"Box coordinates exceed image size at {fname}: {box} : {im_tensor.shape[-2:]}"
                
        return im_tensor, targets, im_info['filename']
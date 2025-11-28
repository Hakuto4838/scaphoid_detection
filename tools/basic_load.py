from PIL import Image
from torchvision.transforms import v2
import PyQt5.QtWidgets as qt
import torch
import matplotlib.pyplot as plt   # 資料視覺化套件
import numpy as np
import os
import torchvision.utils as vutils
import torch.nn as nn
import json



def loadimg(path=None):
    if(path is None):
        path, _  = qt.QFileDialog.getOpenFileName(caption="select image", filter="Images (*.png)")
    if path:
        name = os.path.basename(path)
        return path, Image.open(path), os.path.splitext(name)[0]

def loadDir(path=None):
    images = []
    # names = []
    if(path is None):
        path = qt.QFileDialog.getExistingDirectory()
    if path:
        for filename in os.listdir(path):
            if filename.endswith(('.bmp','.png', '.jpg', '.jpeg')):
                imginfo = {}
                imginfo['image'] = Image.open(os.path.join(path, filename))
                imginfo['path'] = os.path.join(path, filename)
                imginfo['name'] = os.path.splitext(filename)[0]
                images.append(imginfo)
    return images


def print_bar(label, data, title=''):
    plt.clf()
    values = data.cpu().detach().numpy()
    bars = plt.bar(label, values)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,          
            f'{value:.3f}',                   
            ha='center', fontsize=7          
        )

    plt.title(title)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Probability', fontsize=12)

    plt.draw()
    plt.pause(0.1)
    
def load_json_label(labeldir, image_name):
    """
    Returns:
        dict: 讀取到的 JSON 內容，如果檔案不存在或讀取失敗則返回 None
    """
    try:
        # 移除可能的副檔名
        base_name = os.path.splitext(image_name)[0]
        
        # 組合完整的 JSON 檔案路徑
        json_path = os.path.join(labeldir, f"{base_name}.json")
        
        # 檢查檔案是否存在
        if not os.path.exists(json_path):
            print(f"Label file not found: {json_path}")
            return None
            
        # 讀取 JSON 檔案
        with open(json_path, 'r') as f:
            label_data = json.load(f)
            
        return label_data
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
        return None
    except Exception as e:
        print(f"Error loading label file: {e}")
        return None
    
def getdir():
    return qt.QFileDialog.getExistingDirectory()

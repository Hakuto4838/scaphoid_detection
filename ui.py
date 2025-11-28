import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QGroupBox, QSizePolicy, QSpacerItem, QLineEdit, QSlider, QFrame
)
from tools.fasterRCNN import FasterRCNN
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageQt, ImageDraw
import os
import torch
from PyQt5.QtCore import Qt

from tools.basic_load import loadDir, getdir, load_json_label
from tools.tool import get_IOU_score


demo = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mainclass(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Homework 2")
        self.initUI()
        self.net = FasterRCNN()
        self.net.eval()
        self.net.to(device)
        self.net.load_state_dict(torch.load(
                'best_scaphoid_detection.pth',
                map_location = device
            ))
        
        self.imagelst = []
        self.labeldir = None
        self.label2dir = None
        self.len = 0
        self.idx = 0

        self.pred_box = None
        self.label_box = None
        self.pred_score = None
        self.label2_box = None

        self.InputImg = None

    def initUI(self):
        # Main layout
        main_layout = QHBoxLayout(self)
        
        main_layout.addWidget(self.create_control_panel())
        main_layout.addWidget(self.create_image_panel())
        
        # Set stretch factors
        main_layout.setStretch(0, 1)  # Control panel takes 1 part
        main_layout.setStretch(1, 2)  # Image panel takes 2 parts

    def create_control_panel(self):
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)
        
        # Image loading section
        image_group = QGroupBox("Image")
        image_layout = QVBoxLayout(image_group)
        
        btn_load = QPushButton("Load Folder")
        btn_prev = QPushButton("Previous")
        btn_next = QPushButton("Next")
        btn_loadlabel = QPushButton("Load label")
        btn_loadlabel2 = QPushButton("Load label2")
        
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        
        image_layout.addWidget(btn_load)
        image_layout.addLayout(nav_layout)
        image_layout.addWidget(btn_loadlabel)
        image_layout.addWidget(btn_loadlabel2)
        
        # Detection section
        detection_group = QGroupBox("Detection")
        detection_layout = QVBoxLayout(detection_group)
        
        btn_detect = QPushButton("Detection")
        self.iou_label = QLabel("IoU:")
        self.accuracy_label = QLabel("Accuracy:")
        self.precision_label = QLabel("Precision:")
        self.recall_label = QLabel("Recall:")
        
        detection_layout.addWidget(btn_detect)
        detection_layout.addWidget(self.iou_label)
        detection_layout.addWidget(self.accuracy_label)
        detection_layout.addWidget(self.precision_label)
        detection_layout.addWidget(self.recall_label)
        
        # Add all sections to main layout
        layout.addWidget(image_group)
        layout.addWidget(detection_group)
        layout.addStretch()
        
        # Connect button events
        btn_load.clicked.connect(self.load_folder)
        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)
        btn_detect.clicked.connect(self.detect_image)
        btn_loadlabel.clicked.connect(self.load_label)
        btn_loadlabel2.clicked.connect(self.load_label2)
        
        return group

    def create_image_panel(self):
        group = QGroupBox("Image Display")
        layout = QVBoxLayout(group)
        
        # Current image label
        self.current_image_label = QLabel("Current Image: None")
        layout.addWidget(self.current_image_label)
        
        # Image display
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 400)
        self.image_display.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        layout.addWidget(self.image_display)
        
        return group

    def load_folder(self):
        self.imagelst = loadDir()
        self.len = len(self.imagelst)
        self.idx = 0
        print(f"Read {self.len} image")
        if self.len>0:
            self.clean_output()
            self.update_image(self.imagelst[self.idx])

    def prev_image(self):
        if self.idx>0:
            self.idx -= 1
            self.clean_output()
            self.update_image(self.imagelst[self.idx])
            self.detect_image()
        
        pass

    def next_image(self):
        if self.idx < self.len-1:
            self.idx += 1
            self.clean_output()
            self.update_image(self.imagelst[self.idx])
            self.detect_image()

    def detect_image(self):
        if self.labeldir is not None:
            label_json = load_json_label(self.labeldir, self.imagelst[self.idx]['name'])
            if label_json is not None:
                self.set_label_box(label_json[0]['bbox'])
        
            if self.label2dir is not None:
                label2_json = load_json_label(self.label2dir, self.imagelst[self.idx]['name'])
                if label2_json is not None:
                    if label2_json[0]['name'] == 'Fracture':
                        self.set_label2_box(label2_json[0]['bbox'])
                    else:
                        self.label2_box = None

        image = self.imagelst[self.idx]['image']
        rpn_output, frcnn_output = self.net(image)
        self.set_pred_box(frcnn_output['boxes'][0])
        self.pred_score = frcnn_output['scores'][0]

        if (self.pred_box is not None) and (self.label_box is not None):
            iou, acc, prec, recall = get_IOU_score(self.label_box, self.pred_box)
            self.accuracy_label

            self.iou_label.setText(f"IoU: {iou:2.2f}")
            self.accuracy_label.setText(f"Accuracy: {100*acc:2.2f}%")
            self.precision_label.setText(f"Precision: {100*prec:2.2f}%")
            self.recall_label.setText(f"Recall: {100*recall:2.2f}%")
        pass
    
    def clean_output(self):
        self.pred_box = None
        self.pred_score = None
        self.label_box = None
        self.label2_box = None
        self.iou_label.setText(f"IoU: ")
        self.accuracy_label.setText(f"Accuracy:")
        self.precision_label.setText(f"Precision:")
        self.recall_label.setText(f"Recall:")


    def draw_boxes_on_image(self):
        """在圖片上繪製預測框和標籤框"""
        if self.image is None:
            return
        
        if self.image.mode == 'L':
            self.image = self.image.convert('RGB')
        
        # 複製圖片以免修改原圖
        self.image_with_box = self.image.copy()
        draw = ImageDraw.Draw(self.image_with_box)
        
        # 繪製預測框（紅色）
        if (self.pred_box is not None):
            box = self.pred_box.cpu().detach().numpy() if isinstance(self.pred_box, torch.Tensor) else self.pred_box
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline='red',
                width=5
            )
        
        # 繪製標籤框（綠色）
        if (self.label_box is not None):
            box = self.label_box.cpu().detach().numpy() if isinstance(self.label_box, torch.Tensor) else self.label_box
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline='green',
                width=5
            )
        
        # 繪製骨折標籤框（綠色）
        if (self.label2_box is not None):
            points = []
            for point in self.label2_box:
                x = float(point[0].item())
                y = float(point[1].item())
                points.append((x, y))

            draw.polygon(
                points, 
                outline='green',  
                fill=None, 
                width=5  
            )

    def convert_to_pixmap(self, pil_image):
        """將 PIL Image 轉換為 QPixmap"""
        if pil_image is None:
            return None
            
        # PIL Image 轉換為 QImage
        if pil_image.mode == 'RGB':
            r, g, b = pil_image.split()
            im = Image.merge('RGB', (r, g, b))
            data = im.tobytes('raw', 'RGB')
            qim = QImage(data, im.size[0], im.size[1], im.size[0] * 3, QImage.Format_RGB888)
        elif pil_image.mode == 'L':  # 灰階圖片
            data = pil_image.tobytes('raw', 'L')
            qim = QImage(data, pil_image.size[0], pil_image.size[1], pil_image.size[0], QImage.Format_Grayscale8)
            
        return QPixmap.fromImage(qim)

    def display_image(self, show_box=False):
        """在 UI 上顯示圖片"""
        if show_box and self.image_with_box is not None:
            display_image = self.image_with_box
        elif self.image is not None:
            display_image = self.image
        else:
            return
            
        # 轉換為 QPixmap
        pixmap = self.convert_to_pixmap(display_image)
        if pixmap is None:
            return
            
        # 縮放圖片以適應顯示區域
        scaled_pixmap = pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 顯示圖片
        self.image_display.setPixmap(scaled_pixmap)

    def update_image(self, imageinfo):
        """更新圖片並顯示"""
        try:
            self.image = Image.open(imageinfo['path'])
            self.current_image_label.setText(f"[{self.idx+1}/{self.len}]Current Image: {imageinfo['name']}")
            
            # 如果有任一種框，就繪製框
            if not self.pred_box is None or not self.label_box is None : 
                self.draw_boxes_on_image()
                self.display_image(show_box=True)
            else:
                self.display_image(show_box=False)
                
        except Exception as e:
            print(f"Error loading image: {e}")

    def set_pred_box(self, box):
        """設置預測框並更新顯示"""
        # self.pred_box = box
        self.pred_box = list(map(float, box))
        self.pred_box = torch.tensor(self.pred_box, device=device)
        if self.image is not None:
            self.draw_boxes_on_image()
            self.display_image(show_box=True)

    def set_label_box(self, box):
        """設置標籤框並更新顯示"""
        self.label_box = list(map(float, box))
        self.label_box = torch.tensor(self.label_box, device=device)
        if self.image is not None:
            self.draw_boxes_on_image()
            self.display_image(show_box=True)
    
    def set_label2_box(self, box):
        """設置標籤框2並更新顯示"""
        self.label2_box = box
        self.label2_box = torch.tensor(self.label2_box, device=device)
        self.label2_box += self.label_box[:2]
        if self.image is not None:
            self.draw_boxes_on_image()
            self.display_image(show_box=True)

    def load_label(self):
        self.labeldir = getdir()
    def load_label2(self):
        self.label2dir = getdir()
    
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainclass()
    window.show()
    sys.exit(app.exec())


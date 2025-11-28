import torch
import argparse
import os
import numpy as np
import yaml
import random
from tools.fasterRCNN import FasterRCNN
from tqdm import tqdm
from tools.dataloader import ScaphoidDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = 'config'

def visualize_prediction(image_tensor, boxes, save_path):
    """
    視覺化預測結果
    """
    image = T.ToPILImage()(image_tensor.squeeze(0))
    
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def evaluate_model(model, test_loader, device, epoch, save_dir):
    """
    在測試集上評估模型
    """
    test_rpnlosses = []
    test_frcnnlosses = []
    test_losses = []
    visualization_done = False
    
    with torch.no_grad():
        for im, target, fname in tqdm(test_loader, desc='Testing'):
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            
            # 前向傳播
            rpn_output, frcnn_output = model(im, target)
            if not visualization_done:
                v_image = im
                visualization_done = True  

            # 計算損失
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_cls_loss'] + frcnn_output['frcnn_loc_loss']
            loss = rpn_loss + frcnn_loss

            test_rpnlosses.append(rpn_loss.item())
            test_frcnnlosses.append(frcnn_loss.item())
            test_losses.append(loss.item())


    print(f"RPN Loss : {np.mean(test_rpnlosses)}, Frcnn Loss : {np.mean(test_frcnnlosses)}, total loss : {np.mean(test_losses)} ")

    model.eval() #視覺化
    with torch.no_grad():
        rpn_output, frcnn_output = model(v_image)
        boxes = frcnn_output['boxes'][:1].cpu().numpy()
        visualize_prediction(
            v_image.cpu(),
            boxes,
            os.path.join(save_dir, f'epoch_{epoch}_prediction.png')
        )
    model.train()

    return np.mean(test_losses) if test_losses else 0

def train():
    # 讀取配置文件
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    
    # 設定隨機種子以確保可重複性
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 創建訓練資料集
    train_dataset = ScaphoidDataset(
        split='train',
        im_dir=dataset_config['im_train_path'],
        ann_dir=dataset_config['ann_train_path'],
        contrast_range=dataset_config.get('contrast_range', (0.8, 1.2))  # 可選的對比度範圍
    )
    # 創建測試數據集
    test_dataset = ScaphoidDataset(
        split='test',
        im_dir=dataset_config['im_test_path'],
        ann_dir=dataset_config['ann_test_path']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 加速數據傳輸到GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 創建保存視覺化結果的目錄
    vis_dir = os.path.join(train_config['task_name'], 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 初始化模型
    faster_rcnn_model = FasterRCNN(
        num_classes=2  # 背景 + Scaphoid
    )
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    # 創建保存檢查點的目錄
    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'])
    
    # 初始化優化器
    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        lr=train_config['lr'],
        momentum=0.9,
        weight_decay=5E-4
    )
    
    # 學習率調整器
    scheduler = MultiStepLR(
        optimizer, 
        milestones=train_config['lr_steps'],
        gamma=0.1
    )
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1
    best_loss = float('inf')
    
    # 訓練迴圈
    epoch_rpn_cls_loss = []
    epoch_rpn_loc_loss = []
    epoch_frcnn_cls_loss = []
    epoch_frcnn_loc_loss = []
    epoch_total_loss = []
    epoch_test_loss =[]

    # 初始化動態繪圖
    plt.ion()  # 啟用交互模式
    fig = plt.figure(figsize=(10, 5))
    ax_loss = fig.add_subplot(1, 2, 1)
    # ax_acc = fig.add_subplot(1, 2, 2)

    for epoch in range(num_epochs):
        rpn_cls_losses = []
        rpn_loc_losses = []
        frcnn_cls_losses = []
        frcnn_loc_losses = []
        total_losses = []
        
        optimizer.zero_grad()
        
        # 使用tqdm顯示進度條
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for im, target, fname in pbar:
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            
            # 前向傳播
            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            
            # 計算損失
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_cls_loss'] + frcnn_output['frcnn_loc_loss']
            loss = rpn_loss + frcnn_loss
            
            # 記錄損失
            rpn_cls_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_loc_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_cls_losses.append(frcnn_output['frcnn_cls_loss'].item())
            frcnn_loc_losses.append(frcnn_output['frcnn_loc_loss'].item())
            total_losses.append(loss.item())
            
            # 梯度累積
            loss = loss / acc_steps
            loss.backward()
            
            # 更新權重
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            step_count += 1
            
            # 更新進度條資訊
            pbar.set_postfix({
                'total_loss': f'{np.mean(total_losses):.4f}'
            })
        
        # epoch結束，進行最後一次優化步驟
        optimizer.step()
        optimizer.zero_grad()
        
        # 計算平均損失
        mean_total_loss = np.mean(total_losses)
        
        # 保存最佳模型
        if mean_total_loss < best_loss:
            best_loss = mean_total_loss
            torch.save(
                faster_rcnn_model.state_dict(),
                os.path.join(train_config['task_name'], 'best_' + train_config['ckpt_name'])
            )
        
        # # 定期保存檢查點
        # if (epoch + 1) % train_config['save_interval'] == 0:
        #     torch.save(
        #         faster_rcnn_model.state_dict(),
        #         os.path.join(train_config['task_name'], f'epoch_{epoch+1}_' + train_config['ckpt_name'])
        #     )
        
        # 輸出損失資訊
        loss_output = f'\nEpoch {epoch+1}/{num_epochs}\n'
        loss_output += f'RPN Classification Loss: {np.mean(rpn_cls_losses):.4f}'
        loss_output += f' | RPN Localization Loss: {np.mean(rpn_loc_losses):.4f}'
        loss_output += f' | FRCNN Classification Loss: {np.mean(frcnn_cls_losses):.4f}'
        loss_output += f' | FRCNN Localization Loss: {np.mean(frcnn_loc_losses):.4f}'
        loss_output += f' | Total Loss: {mean_total_loss:.4f}'
        print(loss_output)
        
        epoch_rpn_cls_loss.append(np.mean(rpn_cls_losses).item())
        epoch_rpn_loc_loss.append(np.mean(rpn_loc_losses).item())
        epoch_frcnn_cls_loss.append(np.mean(frcnn_cls_losses).item())
        epoch_frcnn_loc_loss.append(np.mean(frcnn_loc_losses).item())
        epoch_total_loss.append(mean_total_loss)
        
        # 更新學習率
        scheduler.step()

        print("\nEvaluating on test set...")
        test_loss = evaluate_model(
            faster_rcnn_model,
            test_loader,
            device,
            epoch + 1,
            vis_dir
        )
        print(f'Test Loss: {test_loss:.4f}')
        epoch_test_loss.append(test_loss.item())

        update_plt(ax_loss, total_losses, epoch_test_loss, epoch)

    plt.ioff()  # 關閉交互模式
    plt.show()
    print('Training completed!')
    log = {
        "epoch_rpn_cls_loss" : epoch_rpn_cls_loss,
        "epoch_rpn_loc_loss" : epoch_rpn_loc_loss,
        "epoch_frcnn_cls_loss" : epoch_frcnn_cls_loss,
        "epoch_frcnn_loc_loss" : epoch_frcnn_loc_loss,
        "epoch_total_loss" : epoch_total_loss,
        "epoch_test_loss" : epoch_test_loss
    }
    with open(train_config['log_path'], 'w') as f:
        json.dump(log, f, indent=4)


    

def update_plt(ax_loss, train_loss, val_loss, epoch):

    # 繪製損失
    ax_loss.clear()
    ax_loss.plot([i*epoch/len(train_loss) for i in range(len(train_loss))], train_loss, label='Training Loss', color='blue')
    ax_loss.plot(range(1,1+len(val_loss)), val_loss, label='Validation Loss', color='orange')
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    # # 繪製準確率
    # ax_acc.clear()
    # ax_acc.plot([i*epoch/len(train_acc) for i in range(len(train_acc))], train_acc, label='Training Accuracy', color='green')
    # ax_acc.plot(range(1,1+len(val_loss)), val_acc, label='Validation Accuracy', color='red')
    # ax_acc.set_title("Accuracy")
    # ax_acc.set_xlabel("Epochs") 
    # ax_acc.set_ylabel("Accuracy")
    # ax_acc.legend()

    # 暫停更新
    plt.pause(0.01)


if __name__ == '__main__':
    train()
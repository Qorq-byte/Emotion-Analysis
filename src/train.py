import os
os.environ['PYTHONUSEV3'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import config
from dataset import get_dataloader
from model import build_model


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='训练中')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, acc, f1


def val_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='验证中'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, acc, f1


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    train_loader = get_dataloader(train=True)
    val_loader = get_dataloader(train=False)
    
    model = build_model().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_f1 = 0
    
    print(f'\n开始训练，共 {config.EPOCHS} 轮...\n')
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f'=== Epoch {epoch}/{config.EPOCHS} ===')
        
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = val_epoch(model, val_loader, criterion, device)
        
        print(f'训练集 - Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f}')
        print(f'验证集 - Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), config.MODELS_DIR / 'best_model.pt')
            print(f'[BEST] 保存最佳模型 (F1: {best_f1:.4f})')
        
        print()
    
    print(f'训练完成! 最佳验证 F1: {best_f1:.4f}')


if __name__ == '__main__':
    train()

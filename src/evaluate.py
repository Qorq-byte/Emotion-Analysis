import os
os.environ['PYTHONUSEV3'] = '0'

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

import config
from dataset import get_dataloader
from model import build_model


def evaluate_model(model_path=None):
    if model_path is None:
        model_path = config.MODELS_DIR / 'best_model.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'加载模型: {model_path}')
    
    test_loader = get_dataloader(train=False)
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print('\n开始评估...\n')
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='评估中'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))
    
    print('\n' + '=' * 60)
    print('[*] 模型评估报告')
    print('=' * 60)
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f'\n[+] 总体指标:')
    print(f'  准确率 (Accuracy):  {acc:.4f}')
    print(f'  精确率 (Precision): {precision:.4f}')
    print(f'  召回率 (Recall):    {recall:.4f}')
    print(f'  F1 值:              {f1:.4f}')
    
    print(f'\n[+] 混淆矩阵:')
    cm = confusion_matrix(all_labels, all_preds)
    print(f'                预测负面  预测正面')
    print(f'  实际负面      {cm[0][0]:>6d}    {cm[0][1]:>6d}')
    print(f'  实际正面      {cm[1][0]:>6d}    {cm[1][1]:>6d}')
    
    print(f'\n[+] 详细分类报告:')
    print(classification_report(all_labels, all_preds, target_names=['负面', '正面']))
    
    print('=' * 60)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def error_analysis(model_path=None, num_examples=5):
    if model_path is None:
        model_path = config.MODELS_DIR / 'best_model.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import pandas as pd
    test_df = pd.read_json(config.PROCESSED_DATA_DIR / 'test.jsonl', lines=True)
    
    test_loader = get_dataloader(train=False)
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    from my_tokenizer import JiebaTokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    
    all_errors = []
    idx = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy().astype(int)
            
            for i in range(len(inputs)):
                if preds[i] != labels_np[i]:
                    encoded = inputs[i].cpu().numpy()
                    tokens = [tokenizer.index2word.get(idx, '<unk>') for idx in encoded]
                    tokens = [t for t in tokens if t not in ['<pad>', '<unk>']]
                    text = ''.join(tokens)
                    
                    all_errors.append({
                        'index': idx,
                        'text': text[:50] + '...' if len(text) > 50 else text,
                        'true_label': '负面' if labels_np[i] == 0 else '正面',
                        'pred_label': '负面' if preds[i] == 0 else '正面',
                        'prob': probs[i]
                    })
                idx += 1
    
    print(f'\n[ERROR] 错误分析 (共 {len(all_errors)} 个错误样本):')
    print('-' * 80)
    for err in all_errors[:num_examples]:
        prob = err['prob'] if err['pred_label'] == '正面' else 1 - err['prob']
        print(f"  真实: {err['true_label']:2s} -> 预测: {err['pred_label']:2s} (置信度: {prob:.1%})")
        print(f"  文本: {err['text']}")
        print()
    
    print(f'... 共 {len(all_errors)} 个错误样本')
    
    return all_errors


if __name__ == '__main__':
    evaluate_model()
    
    print('\n\n')
    error_analysis(num_examples=5)

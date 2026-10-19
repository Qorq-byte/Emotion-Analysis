import argparse

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

import config
from checkpoint import load_model
from dataset import get_dataloader


def evaluate_model(model_path=None, num_examples=5):
    model, tokenizer, metadata = load_model(model_path)
    loader = get_dataloader(train=False, shuffle=False)
    labels, probs = [], []
    with torch.inference_mode():
        for inputs, targets in tqdm(loader, desc="测试集评估"):
            scores = torch.sigmoid(model(inputs.to(metadata["device"])))
            probs.extend(scores.cpu().tolist())
            labels.extend(targets.int().tolist())
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= metadata["threshold"]).astype(int)
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": matrix,
    }
    print(f"测试样本：{len(labels)}；阈值：{metadata['threshold']:.2f}")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key}: {metrics[key]:.4f}")
    print("混淆矩阵（行=真实负/正，列=预测负/正）：")
    print(matrix)
    print(classification_report(labels, preds, labels=[0, 1],
                                target_names=["负面", "正面"], zero_division=0))
    error_indices = np.flatnonzero(preds != labels)
    errors = []
    print(f"错误样本：{len(error_indices)}；展示前 {num_examples} 条")
    for index in error_indices:
        ids = loader.dataset.records[int(index)]["review"]
        tokens = [tokenizer.index2word.get(token_id, "<unk>")
                  for token_id in ids if token_id != tokenizer.pad_index]
        error = {"index": int(index), "text": "".join(tokens),
                 "true_label": "正面" if labels[index] else "负面",
                 "pred_label": "正面" if preds[index] else "负面",
                 "positive_prob": float(probs[index])}
        errors.append(error)
        if len(errors) <= num_examples:
            print(f"  #{index} 真实={error['true_label']} "
                  f"预测={error['pred_label']} "
                  f"正面分数={probs[index]:.1%}  文本片段={error['text'][:80]}")
    print("注意：处理后的数据仅保存 token id，未登录词无法还原为原文。")
    metrics["errors"] = errors
    return metrics


def error_analysis(model_path=None, num_examples=5):
    return evaluate_model(model_path, num_examples)["errors"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在独立测试集评估模型")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    evaluate_model(num_examples=args.examples)

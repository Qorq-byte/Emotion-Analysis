import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm

import config
from checkpoint import vocab_fingerprint
from dataset import ReviewAnalysisDataset, get_dataloader
from model import build_model
from my_tokenizer import JiebaTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, probabilities, labels = [], [], []
    for inputs, targets in tqdm(loader, desc="训练" if training else "验证"):
        inputs, targets = inputs.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()
        losses.append(loss.item() * len(targets))
        probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend(targets.cpu().int().tolist())
    return sum(losses) / len(labels), np.array(probabilities), np.array(labels)


def best_threshold(labels, probabilities):
    candidates = np.arange(0.30, 0.701, 0.01)
    scored = [(f1_score(labels, probabilities >= threshold, zero_division=0),
               -abs(threshold - 0.5), float(threshold)) for threshold in candidates]
    score, _, threshold = max(scored)
    return threshold, score


def train():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = ReviewAnalysisDataset(config.PROCESSED_DATA_DIR / "train.jsonl").records
    if any(len(row["review"]) != config.SEQ_LEN for row in records):
        raise ValueError("数据序列长度与配置不符；请先运行 python src/process.py")
    indices = np.arange(len(records))
    train_idx, val_idx = train_test_split(
        indices, test_size=config.VALIDATION_FRACTION,
        stratify=[row["label"] for row in records], random_state=config.SEED,
    )
    train_loader = get_dataloader(records=[records[i] for i in train_idx], shuffle=True)
    val_loader = get_dataloader(records=[records[i] for i in val_idx], shuffle=False)
    vocab_path = config.MODELS_DIR / "vocab.txt"
    tokenizer = JiebaTokenizer.from_vocab(vocab_path, mode=config.TOKENIZER_MODE)
    model_args = dict(embedding_dim=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE,
                      num_layers=config.NUM_LAYERS, dropout=config.DROPOUT)
    model = build_model(vocab_size=tokenizer.vocab_size, **model_args).to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1, stale = -1.0, 0
    print(f"设备: {device}; 训练: {len(train_idx)}; 验证: {len(val_idx)}; 测试集保持独立")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_prob, train_labels = run_epoch(
            model, train_loader, loss_fn, device, optimizer,
        )
        val_loss, val_prob, val_labels = run_epoch(model, val_loader, loss_fn, device)
        threshold, val_f1 = best_threshold(val_labels, val_prob)
        train_f1 = f1_score(train_labels, train_prob >= 0.5, zero_division=0)
        val_acc = accuracy_score(val_labels, val_prob >= threshold)
        print(f"Epoch {epoch}: train loss={train_loss:.4f} F1={train_f1:.4f}; "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} F1={val_f1:.4f} "
              f"threshold={threshold:.2f}")
        if val_f1 > best_f1 + 1e-6:
            best_f1, stale = val_f1, 0
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(), "model_config": model_args,
                "vocab_sha256": vocab_fingerprint(vocab_path),
                "tokenizer_mode": config.TOKENIZER_MODE, "seq_len": config.SEQ_LEN,
                "threshold": threshold, "validation_f1": val_f1,
                "epoch": epoch, "seed": config.SEED,
            }, config.MODELS_DIR / "best_model.pt")
            print("保存最佳模型")
        else:
            stale += 1
            if stale >= config.PATIENCE:
                print(f"连续 {stale} 轮未改善，提前停止")
                break
    print(f"训练结束；最佳验证 F1={best_f1:.4f}。运行 python src/evaluate.py 评估独立测试集。")


if __name__ == "__main__":
    train()

# LSTM 情感分析

基于 **PyTorch 双向 LSTM** 的中文文本情感分析 demo：对中文商品/服务评论文本进行二分类（正面 / 负面），使用结巴分词构建词表，并提供训练、预测、评估的完整流程。

> 仓库简介：LSTM 制作的情感分析 demo

## 功能特性

- **双向 LSTM 模型**：Embedding + 2 层双向 LSTM + 全连接分类头。
- **中文分词**：使用 jieba 对评论文本分词。
- **二分类情感**：输出正面/负面，并附带 sigmoid 概率与置信度。
- **完整流程**：数据处理 → 词表构建 → 模型训练 → 单条/批量预测 → 详细评估。
- **自动保存最优模型**：以验证集 F1 为指标保存 `best_model.pt`。

## 项目结构

```text
.
├─ src/
│  ├─ config.py        # 路径与训练超参数
│  ├─ my_tokenizer.py  # 基于 jieba 的分词器与词表
│  ├─ dataset.py       # Dataset / DataLoader
│  ├─ model.py         # 双向 LSTM 情感分类模型
│  ├─ train.py         # 训练并保存验证 F1 最优模型
│  ├─ predict.py       # 预测器与示例预测
│  ├─ evaluate.py      # 测试集指标、混淆矩阵、错误分析
│  └─ process.py       # 原始 CSV 数据处理为 jsonl
├─ data/
│  └─ processed/       # 预处理后的训练/测试数据（token id + 标签）
├─ models/
│  ├─ best_model.pt    # 训练好的模型权重
│  └─ vocab.txt        # 结巴分词词表
├─ requirements.txt    # Python 依赖
├─ setup_env.ps1       # PowerShell 虚拟环境激活脚本
└─ logs/               # 日志/运行记录（不入库）
```

## 环境安装

```bash
python -m venv .venv
```

激活虚拟环境（Windows PowerShell）：

```powershell
.\setup_env.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

依赖列表：`torch>=2.0.0`、`pandas>=2.0.0`、`numpy>=1.24.0`、`scikit-learn>=1.3.0`、`tqdm>=4.65.0`、`jieba>=0.42.0`。

## 使用方式

### 直接预测（加载已有模型）

```bash
python src/predict.py
```

脚本会加载 `models/best_model.pt` 与 `models/vocab.txt`，对内置的几条示例评论进行情感预测并输出概率与置信度。

也可以在代码中直接使用：

```python
from src.predict import SentimentPredictor

predictor = SentimentPredictor()
result = predictor.predict("这个商品质量很好，物流也快，下次还买！")
print(result)
```

### 重新训练

```bash
python src/train.py
```

训练过程会输出每轮的 Loss、准确率与 F1，并在验证集 F1 提升时覆盖保存 `models/best_model.pt`。

### 评估模型

```bash
python src/evaluate.py
```

输出测试集的准确率、精确率、召回率、F1、混淆矩阵与分类报告，并展示前若干个错误分类样本。

### 重新处理原始数据

如需要从原始数据重新生成训练/测试集：

1. 将原始数据集文件命名为 `online_shopping_10_cats.csv`，放入 `data/raw/` 目录；
2. 数据集需包含 `label`（0/1）与 `review`（评论文本）两列；
3. 执行：

```bash
python src/process.py
```

脚本会随机采样 10% 数据，按 8:2 分层划分训练/测试集，构建词表并生成 `data/processed/train.jsonl` 与 `test.jsonl`。

> 说明：仓库未包含原始评论文本数据集，仅包含预处理后的 token id 序列；如需复现数据处理请自行准备原始数据。

## 模型与训练配置

主要参数位于 `src/config.py`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `SEQ_LEN` | 128 | 序列长度（截断/填充） |
| `BATCH_SIZE` | 64 | 批大小 |
| `EMBEDDING_DIM` | 128 | 词向量维度 |
| `HIDDEN_SIZE` | 256 | LSTM 隐藏层维度 |
| `LEARNING_RATE` | 1e-3 | 学习率 |
| `EPOCHS` | 10 | 训练轮数 |

模型结构：

- Embedding → 2 层双向 LSTM（dropout 0.5）→ 拼接双向末层 hidden → Dropout + Linear + ReLU → 输出 1 维 logits；
- 训练损失：`BCEWithLogitsLoss`；
- 预测判断：`sigmoid(output) > 0.5` 为正面，否则为负面。

## 隐私与版本控制

- 仓库不包含 API Key、密码等敏感信息。
- `.venv/`（虚拟环境，体积可达数百 MB）、IDE 配置、Python 缓存、日志等文件已通过 `.gitignore` 排除，不会上传。
- 原始评论文本未入库，`data/processed/` 中保存的是分词后的 token id 与标签。


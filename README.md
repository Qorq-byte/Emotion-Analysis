# 中文评论情感分析（双向 LSTM）

对商品和服务评论进行正面/负面二分类。项目包含原始数据处理、训练、独立测试集评估和单条或批量预测。

## 安装

建议使用 Python 3.10–3.13 创建全新环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

仓库已有 `models/best_model.pt`、`models/vocab.txt` 和处理后的数据，可以直接预测和评估。若现有 `.venv` 指向已删除的 Python，请重建虚拟环境。

## 直接使用

```powershell
python src/predict.py "这个商品质量很好，物流也快" "用了两天就坏了"
python src/evaluate.py --examples 5
```

预测会输出正负分数、判定阈值、切词结果、未登录词与截断数。分数未经概率校准，不能视作真实的正确率；切词展示的是模型输入，并非模型解释。旧权重仍可使用，默认采用原来的 jieba 切词和 0.5 阈值。

Python 调用：

```python
from predict import SentimentPredictor

predictor = SentimentPredictor()
result = predictor.predict("这个商品质量很好")
print(result["sentiment"], result["positive_prob"])
```

在交互式 Python 中调用前，先将项目的 `src` 目录加入 `PYTHONPATH`；也可以直接使用上述命令行脚本。

## 训练和数据

```powershell
python src/train.py
python src/evaluate.py
```

训练只读取 `data/processed/train.jsonl`，从中按标签分层拆出 20% 验证集。保存验证 F1 最好的权重，连续 4 轮没有提高则早停。训练完再用 `data/processed/test.jsonl` 做最终评估。旧版本曾用测试集选权重，因此旧权重的测试指标可能偏乐观；重新训练可得到更可靠的评估。

若需要改变切词方式或重新构建词表，先将包含 `label`（0/1）与 `review`（文本）列的 `online_shopping_10_cats.csv` 放到 `data/raw/`，再执行：

```powershell
python src/process.py
python src/train.py
python src/evaluate.py
```

`process.py` 会清理空评论和重复文本，按标签分层抽样 10%，再分为训练与独立测试数据。词表只从训练部分构建，按词频排序并设最小词频与最大规模。**重新处理会覆盖词表和处理后的数据，必须立即重新训练；此前的权重不能再与新词表搭配。** 仓库未提供原始 CSV，因此只凭现有 token id 数据无法重建词表或切换到字级切词。

## 参数与切词

主要设置位于 `src/config.py`：

| 参数 | 默认 | 用途 |
| --- | ---: | --- |
| `TOKENIZER_MODE` | `jieba` | 中文词级切词；可改为 `char` 做字级切词 |
| `SEQ_LEN` | 128 | 最大 token 数；不足部分补零 |
| `MIN_TOKEN_FREQ` / `MAX_VOCAB_SIZE` | 2 / 30000 | 词表过滤与规模 |
| `EMBEDDING_DIM` / `HIDDEN_SIZE` | 128 / 128 | 词向量与 LSTM 隐藏维度 |
| `NUM_LAYERS` / `DROPOUT` | 1 / 0.3 | LSTM 层数与分类头 dropout |
| `BATCH_SIZE` / `LEARNING_RATE` | 64 / 3e-4 | 批大小与 AdamW 学习率 |
| `WEIGHT_DECAY` / `GRAD_CLIP_NORM` | 1e-4 / 1.0 | 权重衰减与梯度裁剪 |
| `EPOCHS` / `PATIENCE` | 20 / 4 | 最大轮数与早停耐心值 |

新模型采用双向 LSTM，跳过填充位置，以免反向 LSTM 把填充位当成评论内容。验证阶段在 0.30–0.70 范围选择 F1 最佳的阈值并随权重保存。新权重会记录模型参数、切词方式、序列长度及词表 SHA-256，预测时检查词表是否匹配。旧权重沿用原始结构和计算路径。

## 数据与指标的边界

- 仓库提供 5021 条训练数据和 1256 条测试数据，均为 token id；没有原始文本。错误样本中的文本只能由词表还原，未登录词显示为 `<unk>`。
- 标签仅有正面和负面。对中性、讽刺、多方面或超出训练领域的文本，输出仍会强制二选一。
- 测试指标应在训练结束后查看一次。若反复基于测试结果调参数，需再准备新的独立测试集。

import argparse

import torch

from checkpoint import load_model


class SentimentPredictor:
    def __init__(self, model_path=None):
        self.model, self.tokenizer, self.metadata = load_model(model_path)
        self.device = self.metadata["device"]

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请输入非空评论文本")
        tokens = self.tokenizer.tokenize(text)
        visible = tokens[:self.metadata["seq_len"]]
        unknown = [token for token in visible
                   if token not in self.tokenizer.word2index]
        encoded = self.tokenizer.encode(text, self.metadata["seq_len"])
        inputs = torch.tensor([encoded], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            positive = torch.sigmoid(self.model(inputs)).item()
        threshold = self.metadata["threshold"]
        positive_label = positive >= threshold
        return {
            "text": text,
            "sentiment": "正面" if positive_label else "负面",
            "positive_prob": positive,
            "negative_prob": 1 - positive,
            "confidence": positive if positive_label else 1 - positive,
            "threshold": threshold,
            "tokens": visible,
            "unknown_tokens": unknown,
            "truncated_tokens": max(0, len(tokens) - len(visible)),
            "tokenizer_mode": self.tokenizer.mode,
        }


def format_result(result):
    # Model scores are not calibrated probabilities and token lists are not explanations.
    return (f"评论：{result['text']}\n"
            f"判断：{result['sentiment']}（正面分数 {result['positive_prob']:.1%}，"
            f"负面分数 {result['negative_prob']:.1%}；判定阈值 {result['threshold']:.2f}）\n"
            f"切词（{result['tokenizer_mode']}）：{' / '.join(result['tokens']) or '无'}\n"
            f"未登录词：{len(result['unknown_tokens'])} 个"
            + (f"（{' / '.join(result['unknown_tokens'][:10])}）" if result['unknown_tokens'] else "")
            + f"；截断：{result['truncated_tokens']} 个 token\n"
            "说明：分数是模型输出，未经概率校准；切词仅展示模型输入，不代表情感依据。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文评论情感预测")
    parser.add_argument("texts", nargs="*", help="待预测的评论；留空使用内置示例")
    args = parser.parse_args()
    examples = args.texts or [
        "这个商品质量很好，物流也快，下次还买！",
        "垃圾东西，用了一天就坏了，差评！",
        "这本书内容一般，不算好也不算坏",
        "客服态度很差，再也不买这家了",
        "太喜欢了，性价比很高，强烈推荐！",
        "这书的质量很好，下次再也不买了",
    ]
    predictor = SentimentPredictor()
    if predictor.metadata["legacy"]:
        print("提示：当前为旧版权重，使用默认 jieba 切词和 0.5 阈值。")
    for item in examples:
        print(format_result(predictor.predict(item)), end="\n\n")

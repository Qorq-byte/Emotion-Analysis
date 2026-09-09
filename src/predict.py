import os
os.environ['PYTHONUSEV3'] = '0'

import torch
import config
from model import build_model
from my_tokenizer import JiebaTokenizer


class SentimentPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.MODELS_DIR / 'best_model.pt'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
        self.model = build_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
    def predict(self, text):
        encoded = self.tokenizer.encode(text, config.SEQ_LEN)
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()
        
        sentiment = '正面' if prob > 0.5 else '负面'
        confidence = prob if prob > 0.5 else 1 - prob
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_prob': prob
        }


if __name__ == '__main__':
    predictor = SentimentPredictor()
    
    test_texts = [
        '这个商品质量很好，物流也快，下次还买！',
        '垃圾东西，用了一天就坏了，差评！',
        '这本书内容一般，不算好也不算坏',
        '客服态度很差，再也不买这家了',
        '太喜欢了，性价比很高，强烈推荐！',
        '这书的质量很好，下次再也不买了'
    ]
    
    print('=' * 60)
    print('LSTM 情感分析测试')
    print('=' * 60)
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f'\n文本: {result["text"]}')
        print(f'情感: {result["sentiment"]}  |  置信度: {result["confidence"]:.2%}')
    
    print('\n' + '=' * 60)

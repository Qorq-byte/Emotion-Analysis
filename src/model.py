import os
os.environ['PYTHONUSEV3'] = '0'

import torch
import torch.nn as nn
import config
from my_tokenizer import JiebaTokenizer


class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes=1, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        embed = self.embedding(x)
        
        output, (hidden, cell) = self.lstm(embed)
        
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        logits = self.fc(hidden_cat)
        
        return logits.squeeze(-1)


def build_model():
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    model = LSTMSentiment(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE
    )
    return model


if __name__ == '__main__':
    model = build_model()
    print(model)
    x = torch.randint(0, 1000, (2, config.SEQ_LEN))
    print('输入形状:', x.shape)
    print('输出形状:', model(x).shape)

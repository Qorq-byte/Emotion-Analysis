import torch
import torch.nn as nn

import config
from my_tokenizer import JiebaTokenizer


class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes=1,
                 num_layers=2, dropout=0.5, use_packing=True):
        super().__init__()
        self.use_packing = use_packing
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        embedded = self.embedding(x)
        if self.use_packing:
            lengths = x.ne(self.embedding.padding_idx).sum(dim=1).clamp(min=1).cpu()
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False,
            )
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(torch.cat([hidden[-2], hidden[-1]], dim=1)).squeeze(-1)


def build_model(vocab_size=None, **kwargs):
    if vocab_size is None:
        vocab_size = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt").vocab_size
    defaults = dict(embedding_dim=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE,
                    num_layers=config.NUM_LAYERS, dropout=config.DROPOUT)
    defaults.update(kwargs)
    return LSTMSentiment(vocab_size, **defaults)

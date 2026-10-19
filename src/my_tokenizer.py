from collections import Counter
from pathlib import Path

import jieba


class JiebaTokenizer:
    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, vocab_list, mode="jieba"):
        if mode not in {"jieba", "char"}:
            raise ValueError("切词形式只能是 jieba 或 char")
        if len(vocab_list) != len(set(vocab_list)):
            raise ValueError("词表含有重复 token")
        if vocab_list[:2] != [self.pad_token, self.unk_token]:
            raise ValueError("词表前两项必须是 <pad>、<unk>")
        self.mode = mode
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = dict(enumerate(vocab_list))
        self.pad_index = 0
        self.unk_index = 1

    def tokenize(self, text):
        if not isinstance(text, str):
            raise TypeError("文本必须是字符串")
        pieces = jieba.lcut(text) if self.mode == "jieba" else list(text)
        return [piece for piece in pieces if piece.strip()]

    def encode(self, text, seq_len):
        if seq_len < 1:
            raise ValueError("seq_len 必须大于 0")
        tokens = self.tokenize(text)[:seq_len]
        indices = [self.word2index.get(token, self.unk_index) for token in tokens]
        return indices + [self.pad_index] * (seq_len - len(indices))

    @classmethod
    def build_vocab(cls, sentences, vocab_path, mode="jieba", min_freq=2, max_size=30000):
        if min_freq < 1 or max_size < 2:
            raise ValueError("min_freq 必须 >= 1 且 max_size 必须 >= 2")
        tokenizer = cls([cls.pad_token, cls.unk_token], mode)
        counts = Counter(token for sentence in sentences for token in tokenizer.tokenize(sentence))
        words = sorted(
            (word for word, count in counts.items()
             if count >= min_freq and word not in {cls.pad_token, cls.unk_token}),
            key=lambda word: (-counts[word], word),
        )[:max_size - 2]
        path = Path(vocab_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([cls.pad_token, cls.unk_token, *words]) + "\n", encoding="utf-8")
        return cls([cls.pad_token, cls.unk_token, *words], mode)

    @classmethod
    def from_vocab(cls, vocab_path, mode="jieba"):
        return cls(Path(vocab_path).read_text(encoding="utf-8").splitlines(), mode)

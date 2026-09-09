import jieba
from tqdm import tqdm

import config

class JiebaTokenizer:
    unk_token='<unk>'
    pad_token='<pad>'
    def __init__(self,vocab_list):
      self.vocab_list=vocab_list
      self.vocab_size=len(vocab_list)
      self.word2index={word:index for index,word in enumerate(vocab_list)}
      self.index2word={index:word for index,word in enumerate(vocab_list)}
      self.unk_index=self.word2index[self.unk_token]
      self.pad_index=self.word2index[self.pad_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self,text,seq_len):
        tokens=self.tokenize( text)
        # 截断或填充序列
        if len(tokens)>seq_len:
            tokens=tokens[:seq_len]
        else:
            tokens=tokens+[self.pad_token]*(seq_len-len(tokens))
        return [self.word2index.get(token,self.unk_index) for token in tokens]

    @classmethod
    def build_vocab(cls,sentences,vocab_path):
        vocab_set = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            vocab_set.update(jieba.lcut(sentence))
        vocab_list = [cls.pad_token,cls.unk_token] + [token for token in vocab_set if token.strip()!='']

        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls,vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR/'vocab.txt')

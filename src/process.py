import os
os.environ['PYTHONUSEV3'] = '0'

import pandas as pd
import torch
import config
from sklearn.model_selection import train_test_split
from my_tokenizer import JiebaTokenizer

def process():
    # 读取原始数据
    print('开始处理数据')
    # 读取原始数据
    df = pd.read_csv(config.RAW_DATA_DIR/'online_shopping_10_cats.csv',usecols=['label','review'],encoding='utf-8').dropna().sample(frac=0.1)

    #划分数据集
    train_df,test_df=train_test_split(df,test_size=0.2,stratify=df['label'],random_state=42)

    #构建词表      
    JiebaTokenizer.build_vocab(train_df['review'].tolist(),config.MODELS_DIR/'vocab.txt')

    # 从词表加载分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR/'vocab.txt')

    #计算序列长度
    # print(train_df['review'].apply(lambda x:len(tokenizer.tokenizer(x))).quantile(0.95))

    #构建训练集
    train_df['review']=train_df['review'].apply(lambda x:tokenizer.encode(x,config.SEQ_LEN))

    #保存训练集
    train_df.to_json(config.PROCESSED_DATA_DIR/'train.jsonl',orient='records',lines=True)

    #构建测试集
    test_df['review']=test_df['review'].apply(lambda x:tokenizer.encode(x,config.SEQ_LEN))

    #保存测试集
    test_df.to_json(config.PROCESSED_DATA_DIR/'test.jsonl',orient='records',lines=True)

    # 查看数据前五行
    # print(df.head())

    print('数据处理完成')

if __name__ == '__main__':
    process()
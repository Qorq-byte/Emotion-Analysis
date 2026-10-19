import json

import torch
from torch.utils.data import DataLoader, Dataset

import config


class ReviewAnalysisDataset(Dataset):
    def __init__(self, path=None, records=None):
        if records is None:
            with open(path, encoding="utf-8") as source:
                records = [json.loads(line) for line in source if line.strip()]
        if not records:
            raise ValueError("数据集为空")
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        return (torch.tensor(item["review"], dtype=torch.long),
                torch.tensor(item["label"], dtype=torch.float32))


ReviewAnanlyDataset = ReviewAnalysisDataset


def get_dataloader(train=True, records=None, shuffle=None):
    path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
    dataset = ReviewAnalysisDataset(path=path if records is None else None, records=records)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE,
                      shuffle=train if shuffle is None else shuffle)

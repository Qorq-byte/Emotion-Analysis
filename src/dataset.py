import os
os.environ['PYTHONUSEV3'] = '0'

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class ReviewAnanlyDataset(Dataset):
    def __init__(self,path):
        self.data=pd.read_json(path,lines= True,orient='records').to_dict(orient='records')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        input_tensor=torch.tensor(self.data[index]['review'],dtype=torch.long)
        target_tensor=torch.tensor(self.data[index]['label'],dtype=torch.float)
        return input_tensor,target_tensor


#提供一个获取dataloader的方法
def get_dataloader(train= True):
    path=config.PROCESSED_DATA_DIR/('train.jsonl' if train else 'test.jsonl')
    dataset=ReviewAnanlyDataset( path)
    return DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)


if __name__=='__main__':
    dataloader=get_dataloader()
    print(len(dataloader))
import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class ICUDataset(Dataset):
    def __init__(self, data_path_X, data_path_y, max_len, item_id_set) -> None:
        super(ICUDataset, self).__init__()

        self.data = open(data_path_X, "rt").readlines()
        self.max_len = max_len
        self.labels = np.load(data_path_y)
        
        self.item_id_dict = dict()
        self.item_id_dict[0] = 0
        item_id_set = list(item_id_set)
        for i in range(len(item_id_set)):
            self.item_id_dict[item_id_set[i]] = i+1

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        line = self.data[index].strip()
        tokens = line.split(" ")[:self.max_len]
        if tokens == [""]: tokens = ["0:0:0"]
        if len(tokens) < self.max_len: tokens += ["0:0:0" for _ in range(self.max_len - len(tokens))]
        
        time = np.array(list(map(lambda x: x.split(":")[0], tokens)), dtype=np.double)
        itemid = np.array(list(map(lambda x: self.item_id_dict[int(x.split(":")[1])], tokens)), dtype=int)
        
        valuenum = np.array(list(map(lambda x: x.split(":")[2], tokens)), dtype=np.double)
        

        label = torch.zeros(2)
        label[int(self.labels[index])] = 1

        return {'label': label,
                'time': time,
                'itemid': itemid,
                'valuenum': valuenum}

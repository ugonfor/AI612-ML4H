import numpy as np
import torch.nn as nn
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


import utils
from typing import List
from type_def import ICUSTAY_Entity

import pandas as pd

ICU_DATA: List[ICUSTAY_Entity] = utils.pause("./ICU_DATA_rnn", type='resume')

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from dataloader_rnn import ICUDataset
from torch.utils.data import DataLoader

ITEMID_SET = {223751, 223752, 227341, 227342, 227343, 227344, 223761, 227345, 220179, 220180, 220181, 227346, 223769, 223770, 227367, 223791, 220210, 51, 52, 224828, 220224, 220739, 581, 220235, 223830, 87, 1126, 618, 113, 220277, 646, 223900, 223901, 676, 677, 678, 679, 8368, 5813, 5815, 184, 5817, 5819, 5820, 198, 226512, 211, 723, 226531, 742, 226543, 226544, 8441, 8448, 776, 777, 778, 779, 780, 811, 813, 224054, 224055, 224056, 224057, 224058, 224059, 829, 225087, 225092, 837, 225094, 225103, 225106, 226137, 8547, 8549, 8551, 227688, 8553, 8554, 224641, 220045, 220046, 220047, 220050, 220051, 220052, 224161, 224162, 225698, 224168, 454, 455, 456, 226253, 470, 492, 1529, 1535}


class GRU(nn.Module):
    def __init__(self, d_itemid, hidden_dim, n_layers=1, n_class=2):
        super(GRU, self).__init__()

    
        self.itemid_embedding = nn.Embedding(101, d_itemid)
        self.gru = nn.GRU(1 + d_itemid + 1, 2 * hidden_dim, n_layers, batch_first=True, ) # charttime:itemid:value
        self.outer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        x_time = x['time']
        x_itemid = x['itemid']
        x_val = x['valuenum']
        emb_itemid = self.itemid_embedding(x_itemid)

        x_time = x_time.reshape((10,100,1))
        x_val = x_val.reshape((10,100,1))
        input = torch.cat((x_time, emb_itemid, x_val), dim=2).float()

        out, _ = self.gru(input)
        out = self.outer(out[:,-1,:])
        out = self.classifier(out)
        return out
    
from datetime import datetime, timedelta

def make_dataset_file():
    ICU_DATA.sort(key=lambda x: x.ICUSTAY_id)
    
    TRAIN_ICU_DATA = list(filter(lambda x: x.ICUSTAY_id % 10 not in (8,9) , ICU_DATA)) 
    TEST_ICU_DATA = list(filter(lambda x: x.ICUSTAY_id % 10 in (8,9) , ICU_DATA)) 

    fout_train_x = open("./X_train_rnn.npy", "wt")
    fout_train_y = open("./y_train_rnn.npy", "wt")

    no_chart = 0
    for entity in TRAIN_ICU_DATA:
        if len(entity.chartevnet) == 0:
            no_chart += 1
        else:
            FIRST_TIME = datetime.strptime(entity.chartevnet[0][1], '%Y-%m-%d %H:%M:%S')
            for CHART in entity.chartevnet:
                c_time = datetime.strptime(CHART[1], '%Y-%m-%d %H:%M:%S')
                c_time = c_time - FIRST_TIME
                c_time = str(c_time.total_seconds()/60)

                fout_train_x.write(f'{c_time}:{CHART[0]}:{CHART[2]} ')
        fout_train_x.write("\n")
        fout_train_y.write(f"{entity.label}\n")
    print(no_chart)

    fout_train_x = open("./X_test_rnn.npy", "wt")
    fout_train_y = open("./y_test_rnn.npy", "wt")

    no_chart = 0
    for entity in TEST_ICU_DATA:
        if len(entity.chartevnet) == 0:
            no_chart += 1
        else:
            FIRST_TIME = datetime.strptime(entity.chartevnet[0][1], '%Y-%m-%d %H:%M:%S')
            for CHART in entity.chartevnet:
                c_time = datetime.strptime(CHART[1], '%Y-%m-%d %H:%M:%S')
                c_time = c_time - FIRST_TIME
                c_time = str(c_time.total_seconds()/60)

                fout_train_x.write(f'{c_time}:{CHART[0]}:{CHART[2]} ')
        fout_train_x.write("\n")
        fout_train_y.write(f"{entity.label}\n")
    print(no_chart)
    
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRU(32, 32)
    model.train()
    model.zero_grad()


    EPOCHS = 10
    dataset = ICUDataset('./X_train_rnn.npy', './y_train.npy', 100, ITEMID_SET)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True) 
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        total_batch = 0
        true_pred = 0
        total_loss = 0
        batch_cnt = 0
        for batch in data_loader:
            # data bach
            data = {
                'time': batch['time'].to(device),
                'itemid': batch['itemid'].to(device),
                'valuenum': batch['valuenum'].to(device),
            }

            #label
            labels = batch['label'].to(device)

            # prediction
            pred = model(data)
            
            # loss
            loss = F.binary_cross_entropy_with_logits(pred, labels)
            
            # optimizer 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # for logging
            pred_y = torch.max(pred,1)[1]
            labels = labels.max(1)[1]
            true_pred += (labels == pred_y).sum().item()

            total_loss += loss.item()

            # batch_size
            total_batch += len(batch['time'])
            batch_cnt += 1
            
        # logging
        print(f"[!] Epoch{epoch} | batch{batch_cnt} | Train loss : {total_loss:.4f} | Train Acc : {true_pred/total_batch * 100:.2f}%")
        
if __name__ == "__main__":
    #make_dataset_file()
    train()

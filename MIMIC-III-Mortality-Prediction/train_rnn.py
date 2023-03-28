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

df = pd.read_csv("./filtered_dataset/CHARTEVENTS.csv")
df = df[df['VALUENUM'].isna() == False]
ITEMID_SET = set(df['ITEMID'].value_counts().index)

NORM = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRU(nn.Module):
    def __init__(self, d_itemid, hidden_dim, n_layers=1, n_class=2):
        super(GRU, self).__init__()

    
        self.itemid_embedding = nn.Embedding(len(ITEMID_SET) + 1, d_itemid)
        self.gru = nn.GRU(1 + d_itemid + 1, 2 * hidden_dim, n_layers, batch_first=True, ) # charttime:itemid:value
        self.outer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_class)
    def forward(self, x):
        x_time = x['time']
        batch_size = x_time.shape[0]
        x_itemid = x['itemid']
        x_val = x['valuenum']
        emb_itemid = self.itemid_embedding(x_itemid)

        x_time = x_time.reshape((batch_size,100,1))
        x_val = x_val.reshape((batch_size,100,1))
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


def train(model, epoch, lr, l2):

    total_predict = torch.tensor([]).to(device)
    print(device)
    model = model.to(device)
    model.train()
    model.zero_grad()


    EPOCHS = epoch
    dataset = ICUDataset('./X_train_rnn.npy', './y_train.npy', 100, ITEMID_SET, NORM)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True) 
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2)
    
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
            weight = torch.Tensor([10, 10]).to(device)
            loss = F.binary_cross_entropy_with_logits(pred, labels, weight=weight)
            
            # optimizer 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # for logging
            pred_y = torch.max(pred,1)[1].to(device)

            total_predict = torch.cat((total_predict, pred_y))

            labels = labels.max(1)[1]
            true_pred += (labels == pred_y).sum().item()
        
            total_loss += loss.item()

            # batch_size
            total_batch += len(batch['time'])
            batch_cnt += 1

        # logging
        print(f"[!] Epoch{epoch} | batch{batch_cnt} | Train loss : {total_loss:.4f} | Train Acc : {true_pred/total_batch * 100:.2f}%")
    return model

def test(model, device, dataset = ['./X_test_rnn.npy', './y_test.npy']):
    print(device)
    total_predict = torch.tensor([]).to(device)
    model.eval()
    
    test_dataset = ICUDataset(dataset[0], dataset[1], 100, ITEMID_SET, NORM)
    test_data_loader = DataLoader(test_dataset, batch_size=10,  shuffle=False) 

    true_pred_test = 0
    total_batch_test = 0
    for batch in test_data_loader:
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
        
        # for logging
        pred = torch.sigmoid(pred)
        tmp = torch.Tensor(pred)
        tmp[:,0] = tmp[:,0]*0.225
        pred = tmp

        pred_y = torch.max(pred,1)[1].to(device)
        total_predict = torch.cat((total_predict, pred_y))
        labels = labels.max(1)[1]
        true_pred_test += (labels == pred_y).sum().item()
        total_batch_test += len(batch['time'])
    
    # logging
    print(f"[!] TEST ACCURACY {true_pred_test/total_batch_test *100:.2f}%")
    
    return total_predict


if __name__ == "__main__":
    # make_dataset_file()
    e_dim = 256
    h_dim = 64
    epoch = 15 # 20 좋았음
    lr=1e-3
    l2=1e-5
    print(e_dim, h_dim, epoch, lr, l2)
    
    # model = GRU(e_dim, h_dim,)
    # m = train(model, epoch, lr, l2)
    # utils.save_model(m, './rnn.pth')
    

    # test
    print("!!!?")
    device = 'cuda'
    model = GRU(e_dim, h_dim).to(device)
    utils.load_model(model, './rnn-best.pth')

    
    # train dataset
    
    tot_pred=test(model, device, ['./X_train_rnn.npy', './y_train.npy'])
    print(tot_pred)
    
    pred = np.array(tot_pred.cpu())
    label = np.load('./y_train.npy')
    
    print(roc_auc_score(label, pred))
    print(average_precision_score(label, pred))
    print(pd.DataFrame(pred).value_counts())
    print(pd.DataFrame(label).value_counts())
    print(np.sum(np.abs(label-pred)))
    
    # test dataset
    tot_pred=test(model, device, ['./X_test_rnn.npy', './y_test.npy'])
    
    pred = np.array(tot_pred.cpu())
    label = np.load('./y_test.npy')
    
    print(roc_auc_score(label, pred))
    print(average_precision_score(label, pred))
    print(pd.DataFrame(pred).value_counts())
    print(pd.DataFrame(label).value_counts())
    print(np.sum(np.abs(label-pred)))


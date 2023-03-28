from typing import List, Dict
from type_def import ICUSTAY_Entity

import pickle
import torch
import os
import random
import time
import numpy as np

def hadm_id_dict(ICU_DATA: List[ICUSTAY_Entity]) -> Dict[int, ICUSTAY_Entity]:
    data = dict()
    for entity in ICU_DATA:
        if entity.HADM_id in data:
            data[entity.HADM_id].append(entity)
        else:
            data[entity.HADM_id] = [entity]
    return data

def icustay_id_dict(ICU_DATA: List[ICUSTAY_Entity]) -> Dict[int, ICUSTAY_Entity]:
    data = dict()
    for entity in ICU_DATA:
        assert entity.ICUSTAY_id not in data
        data[entity.ICUSTAY_id] = entity
    return data

def icustay_id_time_dict_write(ICU_DATA: List[ICUSTAY_Entity], path: str):
    data = dict()
    for entity in ICU_DATA:
        assert entity.ICUSTAY_id not in data
        data[entity.ICUSTAY_id] = entity.INTIME

    with open(path, 'wt') as fout:
        for key in data:
            fout.write(f'{key},{data[key].decode()}\n')


def pause(filename, type='stop', data=None):
    assert type in ('stop', 'resume')
    if type == 'stop':
        with open(f'{filename}.pkl', 'wb') as fout:
            pickle.dump(data, fout)
            return None
    else:
        with open(f'{filename}.pkl', 'rb') as fin:
            return pickle.load(fin)
        

def set_seed(seed):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_model(model, model_path):
    state = {
        'model': model.state_dict()
    }
    torch.save(state, model_path)


def load_model(model, model_path):
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
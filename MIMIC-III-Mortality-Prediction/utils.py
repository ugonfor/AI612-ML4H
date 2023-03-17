from typing import List, Dict
from type_def import ICUSTAY_Entity

import pickle

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

def icustay_id_time_dict_write(ICU_DATA: List[ICUSTAY_Entity]):
    data = dict()
    for entity in ICU_DATA:
        assert entity.ICUSTAY_id not in data
        data[entity.ICUSTAY_id] = entity.INTIME

    for key in data:
        print(f'{key},{data[key].decode()}')


def pause(filename, type='stop', data=None):
    assert type in ('stop', 'resume')
    if type == 'stop':
        with open(f'{filename}.pkl', 'wb') as fout:
            pickle.dump(data, fout)
            return None
    else:
        with open(f'{filename}.pkl', 'rb') as fin:
            return pickle.load(fin)
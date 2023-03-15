from typing import List, Dict
from type_def import ICUSTAY_Entity

def hadm_id_dict(ICU_DATA: List[ICUSTAY_Entity]) -> Dict[int, ICUSTAY_Entity]:
    data = dict()
    for entity in ICU_DATA:
        if entity.HADM_id in data:
            data[entity.HADM_id].append(entity)
        else:
            data[entity.HADM_id] = [entity]
    return data
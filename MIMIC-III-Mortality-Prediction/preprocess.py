import numpy as np
from typing import List, Set
from type_def import ICUSTAY_Entity
from datetime import datetime, timedelta

def filtering_time(input_path: str, output_path: str):
    '''
    filtering the ICUSTAYS Table
    1 <= LOS <= 2

    arg: 
        input_path: ICUSTAYS.cvs file path
        output_path: Filtered ICUSTAYS csv file path
    '''

    fin = open(input_path, "rb")
    fout = open(output_path, "wb")

    # for pass the first line (csv file)
    fout.write(fin.readline())

    while 1:
        line = fin.readline().strip()
        
        # check done
        if line == b"":
            break
        
        los = line.split(b",")[-1]
        
        # check los exist?
        if los == b"":
            continue

        los = float(los)
        if los < 1 or 2 < los:
            continue
             
        fout.write(line + b"\n")
    
def init_ICUSTAYS(input_path: str) -> List[ICUSTAY_Entity]:
    '''
    Make Entity Array
    Array have the ICUSTAY Data

    args:
        input_path: filtered ICUSTAYS.csv path
    
    return: 
        List[ICUSTAY_Entity]
    '''
    fin = open(input_path, "rb")
    fin.readline()

    ICU_data :List[ICUSTAY_Entity] = []

    while 1:
        line = fin.readline().strip()

        # check line exist
        if line == b"": break

        data = line.split(b",")
        entity = ICUSTAY_Entity()
        entity.ICUSTAY_id = int(data[3]) # ICUSTAY_ID
        entity.subject_id = int(data[1]) # SUBJECT_ID
        entity.HADM_id = int(data[2]) # HADM_ID
        entity.INTIME = data[9] # INTIME
        entity.OUTTIME = data[10] # OUTTIME
        
        ICU_data.append(entity)

    return ICU_data

def init_ADMISSIONS(input_path: str, icu_data : List[ICUSTAY_Entity]) -> List[ICUSTAY_Entity]:
    '''
    Add some informations about patients from ADMISSIONS.csv

    args:
        input_path: ADMISSIONS.csv path
        icu_data: ICUSTAY_entity list (ft add info to this list)

    return:
        new ICUSTAY_entity list (info added)
    '''

    # collect hadm id
    # use this to filtering ADMISSIONS.csv 
    hadm_ids = set(map(lambda x: x.HADM_id, icu_data))

    # to use hadm_id - entity pair
    import utils
    hadm_dict = utils.hadm_id_dict(icu_data)
    
    # ICUSTAY_entity list (New)
    new_ICUSTAY_entity = []

    # ADMISSIONS.csv
    fin = open(input_path, "rb")

    # for pass the first line (csv file)
    fin.readline()

    while 1:
        line = fin.readline().strip()

        # check line exist
        if line == b"": break

        data = line.split(b",")
        if int(data[2]) not in hadm_ids: # check hadm ids
            continue
        
        for entity in hadm_dict[int(data[2])]:
            entity.ADMITTIME = data[3]
            entity.DEATHTIME = None if data[5] == b"" else data[5]
            entity.ADMISSION_TYPE = data[6]

            new_ICUSTAY_entity.append(entity)

    return new_ICUSTAY_entity

def Label_check(icu_data : List[ICUSTAY_Entity]) -> List[ICUSTAY_Entity]:

    # If death occur before the INTIME, exclude that entity
    exclude = []

    for i in range(len(icu_data)):
        entity = icu_data[i]

        # if dead
        if entity.DEATHTIME != None:
            IN = datetime.strptime(entity.INTIME.decode(), '%Y-%m-%d %H:%M:%S')
            OUT = datetime.strptime(entity.OUTTIME.decode(), '%Y-%m-%d %H:%M:%S')
            DEA = datetime.strptime(entity.DEATHTIME.decode(), '%Y-%m-%d %H:%M:%S')

            # If death occur before the INTIME, exclude this
            if DEA < IN:
                exclude.append(i)
                continue
            
            # labeling
            if OUT < DEA:
                entity.label = 0
            else:
                entity.label = 1

        # if alived
        else:
            entity.label = 0
    
    # for avoid indexing error, [::-1] 
    for num in exclude[::-1]:
        del icu_data[num]
    
    return icu_data
    
def Filtering_CHARTEVENTS(input_path: str, output_path: str, icu_data: List[ICUSTAY_Entity]):
    '''
    filtering the CHARTEVENTS table
    opt 1. ICUSTAY_ID in icu_data
    opt 2. CHARTTIME <= ICUSTAY_ENTITY.INTIME + 3hours

    arg: 
        input_path: CHARTEVENTS.cvs file path
        output_path: Filtered CHARTEVENTS csv file path
    '''

    # collect icustay_ids
    # use this to filtering CHARTEVENTS.csv 
    ICUSTAY_ID_SET = set(map(lambda x: x.ICUSTAY_id, icu_data))

    # to use icustay - entity pair
    import utils
    icustay_dict = utils.icustay_id_dict(icu_data)
    for key in icustay_dict:
        tmp = icustay_dict[key].INTIME
        icustay_dict[key] = datetime.strptime(tmp.decode(), '%Y-%m-%d %H:%M:%S')
        

    fin = open(input_path, "rb")
    fout = open(output_path, "wb")

    # for pass the first line (csv file)
    fout.write(fin.readline())

    # for progress
    idx = 0
    while 1:
        idx += 1
        if idx % 10000 == 0:
            print(idx)

        line = fin.readline().strip()
        
        # check done
        if line == b"":
            break
        
        # check ICUSTAY_ID exist?
        ICUSTAY_ID = line.split(b",")[3]
        
        if ICUSTAY_ID == b"":
            continue

        ICUSTAY_ID = int(ICUSTAY_ID)
        if ICUSTAY_ID not in ICUSTAY_ID_SET:
            continue
        
        # check CHARTTIME
        CHARTTIME = line.split(b",")[5]

        if CHARTTIME == b"":
            continue

        CHARTTIME = datetime.strptime(CHARTTIME.decode(), '%Y-%m-%d %H:%M:%S')
        INTIME = icustay_dict[ICUSTAY_ID]

        if INTIME + timedelta(hours=3) < CHARTTIME:
            continue
        
        fout.write(line + b"\n")

def init_CHARTEVENTS(icu_data : List[ICUSTAY_Entity]) -> List[ICUSTAY_Entity]:
    pass

def test():
    N = 10
    M = 20
    x = np.zeros((N,M))
    print(x.shape)

    with open('test.npy', 'wb') as f:
        np.save(f, x)

    print(np.load('test.npy'))


import sys
ICUSTAYS_PATH = "./dataset/ICUSTAYS.csv"
FILTERED_ICUSTAYS_PATH = "./filtered_dataset/ICUSTAYS.csv"
ADMISSIONS_PATH = "./dataset/ADMISSIONS.csv"
CHAREVENTS_PATH = "./dataset/CHARTEVENTS.csv"
FILTERED_CHAREVENTS_PATH = "./filtered_dataset/CHARTEVENTS.csv"
ICUSTAY_ID_TIME_PAIR_PATH = "./filtered_dataset/ICUSTAY_ID_TIME_PAIR.csv"
ICU_DATA = None

if __name__ == "__main__":
    # filtering_time(ICUSTAYS_PATH, FILTERED_ICUSTAYS_PATH)
    # print("1")
    # ICU_DATA = init_ICUSTAYS(FILTERED_ICUSTAYS_PATH)
    # print("2")
    # ICU_DATA = init_ADMISSIONS(ADMISSIONS_PATH, ICU_DATA)
    # ICU_DATA = Label_check(ICU_DATA)

    from utils import pause
    ICU_DATA = pause('ICU_DATA', type='resume')

    import utils
    utils.icustay_id_time_dict_write(ICU_DATA, ICUSTAY_ID_TIME_PAIR_PATH)

    # g++ -o chartevents_filter ./chartevents_filter.cpp
    # ./chartevents_filter ./dataset/CHARTEVENTS.csv ./filtered_dataset/CHARTEVENTS.csv ./filtered_dataset/ICUSTAY_ID_TIME_PAIR.csv
    

    # Filtering_CHARTEVENTS(CHAREVENTS_PATH, FILTERED_CHAREVENTS_PATH, ICU_DATA)

    # ICU_DATA = pause('ICU_DATA', type='stop', data=ICU_DATA)
class CHART_EVENT:
    pass



class ICUSTAY_Entity:
    def __init__(self) -> None:
        # from ICUSTAYS
        self.ICUSTAY_id = None # ID
        self.subject_id = None # patient id
        self.HADM_id = None # hospital admission id
        self.label = None # Death/Alive

        self.INTIME = None
        self.OUTTIME = None 

        # from ADMISSONS
        self.ADMITTIME = None
        self.DEATHTIME = None
        self.ADMISSION_TYPE = None
        # there are many more info, but I don't use

        self.chartevnet = [] # CHARTEVENT

    def __str__(self) -> str:
        return f'''
        ICUSTAY_id : {self.ICUSTAY_id} 
        subject_id : {self.subject_id}
        HADM_id : {self.HADM_id}
        label : {self.label}
        INTIME : {self.INTIME}
        OUTTIME : {self.OUTTIME} 
        ADMITTIME : {self.ADMITTIME}
        DEATHTIME : {self.DEATHTIME}
        ADMISSION_TYPE : {self.ADMISSION_TYPE}
        chartevnet : {self.chartevnet} 
'''

    def __repr__(self) -> str:
        return f'''
        ICUSTAY_id : {self.ICUSTAY_id} 
        subject_id : {self.subject_id}
        HADM_id : {self.HADM_id}
        label : {self.label}
        INTIME : {self.INTIME}
        OUTTIME : {self.OUTTIME} 
        ADMITTIME : {self.ADMITTIME}
        DEATHTIME : {self.DEATHTIME}
        ADMISSION_TYPE : {self.ADMISSION_TYPE}
        chartevnet : {self.chartevnet} 
'''
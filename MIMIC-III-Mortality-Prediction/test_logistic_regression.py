import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


import utils
from typing import List
from type_def import ICUSTAY_Entity

import pandas as pd

ICU_DATA: List[ICUSTAY_Entity] = utils.pause("./ICU_DATA_logistic", type='resume')


def test():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    print(X)
    print(y)

def make_X_y(ICU_DATA):
    X = []
    for entity in ICU_DATA:
        array = []
        for ITEMID in entity.chartevnet:
            if entity.chartevnet[ITEMID] == []:
                array.append(0)
            else:
                array.append(np.mean(entity.chartevnet[ITEMID]))
        X.append(array)
    X = np.array(X)

    y = []
    for entity in ICU_DATA:
        y.append(entity.label)
    y = np.array(y)

    return X,y

def predict():
    ICU_DATA.sort(key=lambda x: x.ICUSTAY_id)
    
    TRAIN_ICU_DATA = list(filter(lambda x: x.ICUSTAY_id % 10 not in (8,9) , ICU_DATA)) 
    TEST_ICU_DATA = list(filter(lambda x: x.ICUSTAY_id % 10 in (8,9) , ICU_DATA)) 

    X,y = make_X_y(TRAIN_ICU_DATA)
    np.save('y_train', y)

    clf = LogisticRegression(random_state=0, max_iter=10000).fit(X,y)
    print(clf.score(X,y))
    print(roc_auc_score(y, clf.predict_proba(X)[:,1]))
    print(average_precision_score(y, clf.predict_proba(X)[:,1]))


    print(pd.DataFrame(clf.predict(X)).value_counts())
    print(pd.DataFrame(y).value_counts())

    print("="*50)
    
    X,y = make_X_y(TEST_ICU_DATA)
    np.save('y_test', y)
    
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(X,y)
    print(clf.score(X,y))
    print(roc_auc_score(y, clf.predict_proba(X)[:,1]))
    print(average_precision_score(y, clf.predict_proba(X)[:,1]))


    print(pd.DataFrame(clf.predict(X)).value_counts())
    print(pd.DataFrame(y).value_counts())

    print("="*50)

    
if __name__ == "__main__":
    predict()
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pickle

def predict():
    fout = open("20233477_logistic_regression.txt", 'wt')
    fout.write("20233477\n")

    X_train = np.load('./X_train_logistic.npy')
    y_train = np.load('./y_train.npy')

    fin = open("./logistic.pkl","rb")
    clf = pickle.load(fin)

    fout.write(str(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])))
    fout.write("\n")
    fout.write(str(average_precision_score(y_train, clf.predict_proba(X_train)[:,1])))
    fout.write("\n")
    
    X_test = np.load('./X_test_logistic.npy')
    y_test = np.load('./y_test.npy')
    
    fout.write(str(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))
    fout.write("\n")
    fout.write(str(average_precision_score(y_test, clf.predict_proba(X_test)[:,1])))
    fout.write("\n")

    
if __name__ == "__main__":
    predict()
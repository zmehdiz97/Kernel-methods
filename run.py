from collections import Counter
import os
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from src.pca import PCA
from src.utils import accuracy_score, plot_confusion_matrix
from src.multiclass_svm import OneVSOneMulticlassSVM, OneVSAllMulticlassSVM

def write_submission(Yte, Yte_path):
    assert len(Yte) == 2000
    df = pd.DataFrame(index=np.arange(1, len(Yte) + 1),
                      data=Yte.astype(int),
                      columns=["Prediction"])
    df.to_csv(Yte_path, index_label="Id")

if __name__ == '__main__':
    compare_to_scikit =False
    DATA_DIR = 'data'
    
    Xtr_path = os.path.join(DATA_DIR, "Xtr_hoog.npy")
    Xtr_path_sift = os.path.join(DATA_DIR, "Xtr_sift.npy")

    #Xtro_path = os.path.join(DATA_DIR, "Xtr_hoog.npy")
    Xte_path = os.path.join(DATA_DIR, "Xte_hoog.npy")
    Xte_path_sift = os.path.join(DATA_DIR, "Xte_sift.npy")

    Ytr_path = os.path.join(DATA_DIR, "Ytr.csv")

    Xtr = np.load(Xtr_path)    
    Xte = np.load(Xte_path)
    Xtr_sift = np.load(Xtr_path_sift)    
    Xte_sift = np.load(Xte_path_sift)
    Xtr = np.concatenate((Xtr, Xtr_sift), axis=1)
    Xte = np.concatenate((Xte, Xte_sift), axis=1)

    Ytr_csv = pd.read_csv(Ytr_path).Prediction
    Ytr = np.array(Ytr_csv.tolist())

    print("Loaded image features - shape {}".format(Xtr.shape))

    tic = time.time()
    print("Applying PCA")
    n_components = 180
    pca = PCA(n_components=n_components)
    Xtr = pca.fit(Xtr, scale=True)
    print("Variance explained: {:.2f}".format(
                np.sum(pca.e_values_ratio_)))
    print("PCA applied in {0:.1f}s".format(time.time() - tic))
    Xte = pca.transform(Xte, scale=True)
    
    kf = KFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(kf.split(Xtr)):
        Xtr_, Xval = Xtr[train_index], Xtr[test_index]
        Ytr_, Yval = Ytr[train_index], Ytr[test_index]

    
        print("Training on {} samples, validating on {} samples".format(
                Xtr_.shape[0], Xval.shape[0]))
        print("Features: {}".format(Xtr_.shape[1]))
    
        if False:#compare_to_scikit:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            clf = SVC(kernel='rbf')#, decision_function_shape='ovo')
            clf.fit(Xtr_, Ytr_)
            Ypred = clf.predict(Xval)
            accuracy = accuracy_score(Yval, Ypred)
            print(clf._gamma)
            print("Accuracy: {}".format(accuracy))

        model = OneVSOneMulticlassSVM(kernel='rbf', gamma=0.00091, C=1)
        print(model.gamma)
        model.fit(Xtr_, Ytr_)
        Ypred = model.predict(Xval)
        
        accuracy = accuracy_score(Yval, Ypred)
        print("Accuracy: {}".format(accuracy))
        plot_confusion_matrix(Yval, Ypred)
        Yte = model.predict(Xte)
        print(len(Yte))
        
        Yte_path = os.path.join(DATA_DIR, f"Yte_fol{fold}.csv")
        write_submission(Yte, Yte_path)
        

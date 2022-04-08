import os

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Make sure the directories exist
for directory in [DATA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_confusion_matrix(Yval, Ypred, classes=np.arange(10),
                          normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Reds, directory=DATA_DIR,
                          classifier="no classifier specified"):
    """
    This function prints and plots the confusion matrix.
    It also saves it in a directory called conf_mat
    Normalization can be applied by setting `normalize=True`.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    cm = confusion_matrix(Yval, Ypred)
    saving_dir = os.path.join(directory, "{}_{}.png".format(title, classifier))
    plt.savefig(saving_dir)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def accuracy_score(Yval, Ypred):
    assert len(Yval) == len(Ypred)
    accuracy = np.sum(Yval == Ypred) / len(Yval)
    return accuracy

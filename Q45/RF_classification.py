import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn

from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from einops import rearrange
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from DESCRIPTOR import ready
from utils.visualize import visualize_confusion_matrix, plot_confusion_matrix

if __name__ == '__main__':
    plt.ioff()
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.set_defaults(vis=True)
    # print(sklearn.__version__)
    print(matplotlib.__version__)
    args = parser.parse_args()

    c = 10
    n_images_per_class = 15
    vocabulary_size = 100
    Random_Seed = 0
    classes = os.listdir(os.path.abspath('Caltech_101'))
    CLASS = {'tick': 0,
             'trilobite': 1,
             'umbrella': 2,
             'watch': 3,
             'water_lilly': 4,
             'wheelchair': 5,
             'wild_cat': 6,
             'windsor_chair': 7,
             'wrench': 8,
             'yin_yang': 9}

    dataset, CODEBOOK = ready()

    # ---- Random Forest Classifier ----
    # Change the RF parameters including 
    # the number of trees (n_estimators),  
    # the depth of trees (max_depth),
    # the degree of randomness parameter (bootstrap, max_samples), 
    # the type of weak-learners: e.g. axis-aligned or two-pixel test)
    rf_classifier = RandomForestClassifier(n_estimators=100,
                                           criterion='entropy',
                                           max_depth=None,
                                           min_samples_split=2,
                                           min_samples_leaf=1,
                                           min_weight_fraction_leaf=0.0,
                                           max_features='auto',
                                           max_leaf_nodes=None,
                                           min_impurity_decrease=0.0,
                                           bootstrap=True,
                                           oob_score=False,
                                           n_jobs=1,
                                           random_state=Random_Seed,
                                           verbose=0,
                                           warm_start=False,
                                           class_weight=None,
                                           ccp_alpha=0.0,
                                           max_samples=None)

    # Training
    X = []  # n_images, vocabulary_size
    y = []  # n_images, 1
    for key, value in dataset['train_hist'].items():
        X.append(np.array(value))
        y.append(np.array(CLASS[key]))
    X = rearrange(X, 'c n k -> (c n) k', c=c, n=n_images_per_class, k=vocabulary_size)
    y = np.repeat(y, n_images_per_class)
    rf_classifier.fit(X, y)
    train_score = rf_classifier.score(X, y)
    print(f"Train Accuracy: {train_score}")

    # Testing
    X_test = []  # n_images, vocabulary_size
    for key, value in dataset['test_hist'].items():
        X_test.append(np.array(value))
    X_test = rearrange(X_test, 'c n k -> (c n) k', c=c, n=n_images_per_class, k=vocabulary_size)
    y_pred = rf_classifier.predict(X_test)
    test_score = accuracy_score(y, y_pred)
    print(f"Test Accuracy: {test_score}")

    # Confusion Marix
    cm = confusion_matrix(y, y_pred)
    if args.vis:
        # visualize_confusion_matrix(cm,  title='Recognition_Accuracy_Test')
        plot_confusion_matrix(cm, target_names=classes, title='Recognition_Test')
        # cm = pd.DataFrame(confusion_matrix(y, y_pred), columns=classes, index=classes)
        # sns.heatmap(cm, annot=True)

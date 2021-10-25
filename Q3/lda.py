import warnings
import time
import numpy as np

from scipy import io
from tqdm import trange
from numpy import linalg
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from utils.dataset import split_train_test
from utils.visualize import visualize_confusion_matrix

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.add_argument("--best_m", default=50, type=int, help="Number of best eigen choices")
    # parser.add_argument("--m_pca", default=416 - 52, type=int, help="Dimension in PCA")  # Number of Images - class
    parser.add_argument("--m_pca", default=52 - 1, type=int, help="Dimension in PCA")  # Number of Images - class
    parser.add_argument("--m_lda", default=52 - 1, type=int, help="Dimension in LDA")  # class - 1

    parser.set_defaults(vis=False)

    args = parser.parse_args()

    face_data = io.loadmat('Q3/data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    """ Split Dataset """
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)
    x_train = dataset["train_faces"]
    y_train = dataset["train_identities"]
    x_test = dataset["test_faces"]
    y_test = dataset["test_identities"]

    """ 0. Normalize """
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    """ 1. PCA """
    pca = PCA(n_components=51)
    x_train_pca = pca.fit(x_train).transform(x_train)
    x_test_pca = pca.transform(x_test)

    """ 2. LDA """
    lda = LinearDiscriminantAnalysis(n_components=51)
    x_train_lda = lda.fit(x_train_pca, y_train).transform(x_train_pca)
    x_test_lda = lda.transform(x_test_pca)
    # x_train_lda = lda.fit(x_train, y_train).transform(x_train)
    # x_test_lda = lda.transform(x_test)

    """ 3. NN Classification """
    """ 3.1 Vanilla """
    clf = KNeighborsClassifier(n_neighbors=52)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    print("clf.score             : {0:.3f}".format(clf.score(x_train, y_train)))
    print("clf.score             : {0:.3f}".format(clf.score(x_test, y_test)))
    # print("(pred == y_test) score: {0:.3f}".format((prediction == y_test).mean()))
    visualize_confusion_matrix(y_test, prediction, 'Reconstruction_Vanilla')

    """ 3.2 PCA """
    clf_pca = KNeighborsClassifier(n_neighbors=52)
    clf_pca.fit(x_train_pca, y_train)
    prediction_pca = clf_pca.predict(x_test_pca)
    print("clf.score             : {0:.3f}".format(clf_pca.score(x_train_pca, y_train)))
    print("clf.score             : {0:.3f}".format(clf_pca.score(x_test_pca, y_test)))
    # print("(pred == y_test) score: {0:.3f}".format((prediction_pca == y_test).mean()))
    visualize_confusion_matrix(y_test, prediction_pca, 'Reconstruction_PCA')

    """ 3.3 PCA + LDA """
    clf_lda = KNeighborsClassifier(n_neighbors=52)
    clf_lda.fit(x_train_lda, y_train)
    prediction_lda = clf_lda.predict(x_test_lda)
    print("clf.score             : {0:.3f}".format(clf_lda.score(x_train_lda, y_train)))
    print("clf.score             : {0:.3f}".format(clf_lda.score(x_test_lda, y_test)))
    # print("(pred == y_test) score: {0:.3f}".format(np.mean((prediction_lda == y_test))))
    visualize_confusion_matrix(y_test, prediction_lda, 'Reconstruction_LDA')

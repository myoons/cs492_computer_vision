import warnings
import time
import numpy as np

from scipy import io
from tqdm import trange
from numpy import linalg
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from utils.dataset import split_train_test
from utils.visualize import visualize_face, visualize_faces, visualize_graph, visualize_tsne

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.add_argument("--best_m", default=50, type=int, help="Number of best eigen choices")
    parser.add_argument("--m_pca", default=416 - 52, type=int, help="Dimension in PCA")  # Number of Images - class
    parser.add_argument("--m_lda", default=52 - 1, type=int, help="Dimension in LDA")  # class - 1

    parser.set_defaults(vis=False)

    args = parser.parse_args()

    face_data = io.loadmat('data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    """ Split Dataset """
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)
    x_train = dataset["train_faces"]
    y_train = dataset["train_identities"]
    x_test = dataset["train_faces"]
    y_test = dataset["test_identities"]

    """ 1. PCA """
    pca = PCA(n_components=args.m_pca)
    principal_components = pca.fit_transform(dataset["train_faces"])

    """ 2. LDA """
    lda = LinearDiscriminantAnalysis(n_components=args.m_lda)
    lda.fit(dataset["train_faces"], dataset["train_identities"])
    x_train_lda = lda.transform(x_train)
    x_test_lda = lda.transform(x_test)

    """ 3. NN Classification """
    """ 3.1 PCA """
    clf = KNeighborsClassifier(n_neighbors=52)
    clf.fit(dataset["train_faces"], dataset["train_identities"])
    prediction = clf.predict(dataset["test_faces"])

    print("clf.score             : {0:.3f}".format(clf.score(x_train, y_train)))
    print("(pred == y_test) score: {0:.3f}".format((prediction == y_test).mean()))

    """ 3.2 PCA + LDA """
    clf_lda = KNeighborsClassifier(n_neighbors=52)
    clf_lda.fit(x_train_lda, y_train)
    prediction = clf_lda.predict(x_test_lda)

    print("clf.score             : {0:.3f}".format(clf_lda.score(x_train_lda, y_train)))
    print("(pred == y_test) score: {0:.3f}".format((prediction == y_test).mean()))

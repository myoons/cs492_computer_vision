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
from mpl_toolkits.mplot3d import Axes3D


from utils.dataset import split_train_test
from utils.visualize import visualize_confusion_matrix
from utils.visualize import visualize_faces

warnings.filterwarnings("ignore")


def reconstruction_accuracies(dataset):
    x_train = dataset["train_faces"]
    y_train = dataset["train_identities"]
    x_test = dataset["test_faces"]
    y_test = dataset["test_identities"]
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    test_accuracy = np.zeros((52, 365))
    train_accuracy = np.zeros((52, 365))
    x = np.linspace(1, 365, num=365)
    y = np.linspace(1, 52, num=52)
    x, y = np.meshgrid(x, y)

    for i_lda in range(1, 52, 10):
        for i_pca in range(i_lda, 365, 10):
            """ 1. PCA """
            pca = PCA(n_components=i_pca)
            x_train_pca = pca.fit(x_train).transform(x_train)
            x_test_pca = pca.transform(x_test)
    
            """ 2. LDA """
            lda = LinearDiscriminantAnalysis(n_components=i_lda)
            x_train_lda = lda.fit(x_train_pca, y_train).transform(x_train_pca)
            x_test_lda = lda.transform(x_test_pca)
    
            """ 3.3 PCA + LDA """
            clf_lda = KNeighborsClassifier(n_neighbors=52)
            clf_lda.fit(x_train_lda, y_train)
            train_accuracy[i_lda][i_pca] = clf_lda.score(x_train_lda, y_train)
            test_accuracy[i_lda][i_pca] = clf_lda.score(x_test_lda, y_test)
    max_x = np.argmax(train_accuracy, axis=0)
    max_y = np.argmax(train_accuracy, axis=1)
    max_x_test = np.argmax(test_accuracy, axis=0)
    max_y_test = np.argmax(test_accuracy, axis=1)
    

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, train_accuracy, alpha=0.3, color='blue', edgecolor='blue')
    plt.title('Train')
    plt.xlabel('M_lda')
    plt.ylabel('M_pca')
    plt.savefig('Q3/figures/Train_Reconstruction_Loss.png')
    plt.show()

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, test_accuracy, alpha=0.3, color='blue', edgecolor='blue')
    plt.title('Test')
    plt.xlabel('M_pca')
    plt.ylabel('M_lda')
    plt.savefig('Q3/figures/Test_Reconstruction_Loss.png')
    plt.show()


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
    # face_data = io.loadmat('data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    """ Split Dataset """
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)
    x_train = dataset["train_faces"]
    y_train = dataset["train_identities"]
    x_test = dataset["test_faces"]
    y_test = dataset["test_identities"]
    
    calculation_time = {}

    """ 0. Normalize """
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    

    """ 1. PCA """
    start = time.time()
    pca = PCA(n_components=51)
    x_train_pca = pca.fit(x_train).transform(x_train)
    x_test_pca = pca.transform(x_test)
    calculation_time['pca'] = time.time() - start

    """ 2. LDA """
    start = time.time()
    lda = LinearDiscriminantAnalysis(n_components=51)
    x_train_lda = lda.fit(x_train_pca, y_train).transform(x_train_pca)
    x_test_lda = lda.transform(x_test_pca)
    # x_train_lda = lda.fit(x_train, y_train).transform(x_train)
    # x_test_lda = lda.transform(x_test)
    calculation_time['lda'] = time.time() - start
    
    """ 3. NN Classification """
    """ 3.1 Vanilla """
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=52)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    calculation_time['clf_vanilla'] = time.time() - start
    print("clf.score             : {0:.3f}".format(clf.score(x_train, y_train)))
    print("clf.score             : {0:.3f}".format(clf.score(x_test, y_test)))
    
    success_case = np.where(prediction == y_test)[0]
    fail_case = np.where(prediction != y_test)[0]
    visualize_faces(faces=np.array(x_test[success_case[0:3]]), n=1, title="success_case_vanilla", cols=3, rows=1)
    visualize_faces(faces=np.array(x_test[fail_case[0:3]]), n=1, title="fail_case_vanilla", cols=3, rows=1)

    # print("(pred == y_test) score: {0:.3f}".format((prediction == y_test).mean()))
    # visualize_confusion_matrix(y_test, prediction, 'Reconstruction_Vanilla')
    
    """ 3.2 PCA """
    start = time.time()
    clf_pca = KNeighborsClassifier(n_neighbors=52)
    clf_pca.fit(x_train_pca, y_train)
    prediction_pca = clf_pca.predict(x_test_pca)
    calculation_time['clf_pca'] = time.time() - start
    print("clf.score             : {0:.3f}".format(clf_pca.score(x_train_pca, y_train)))
    print("clf.score             : {0:.3f}".format(clf_pca.score(x_test_pca, y_test)))
    # print("(pred == y_test) score: {0:.3f}".format((prediction_pca == y_test).mean()))
    # visualize_confusion_matrix(y_test, prediction_pca, 'Reconstruction_PCA')
    success_case = np.where(prediction_pca == y_test)[0]
    fail_case = np.where(prediction_pca != y_test)[0]
    visualize_faces(faces=np.array(x_test[success_case[0:3]]), n=1, title="success_case_pca", cols=3, rows=1)
    visualize_faces(faces=np.array(x_test[fail_case[0:3]]), n=1, title="fail_case_pca", cols=3, rows=1)

    """ 3.3 PCA + LDA """
    start = time.time()
    clf_lda = KNeighborsClassifier(n_neighbors=52)
    clf_lda.fit(x_train_lda, y_train)
    prediction_lda = clf_lda.predict(x_test_lda)
    calculation_time['clf_lda'] = time.time() - start
    print("clf.score             : {0:.3f}".format(clf_lda.score(x_train_lda, y_train)))
    print("clf.score             : {0:.3f}".format(clf_lda.score(x_test_lda, y_test)))
    # print("(pred == y_test) score: {0:.3f}".format(np.mean((prediction_lda == y_test))))
    # visualize_confusion_matrix(y_test, prediction_lda, 'Reconstruction_LDA')
    success_case = np.where(prediction_lda == y_test)[0]
    fail_case = np.where(prediction_lda != y_test)[0]
    visualize_faces(faces=np.array(x_test[success_case[0:3]]), n=1, title="success_case_lda", cols=3, rows=1)
    visualize_faces(faces=np.array(x_test[fail_case[0:3]]), n=1, title="fail_case_lda", cols=3, rows=1)

    """ Measure Reconstruction Accuracies """
    # reconstruction_accuracies(dataset=dataset)

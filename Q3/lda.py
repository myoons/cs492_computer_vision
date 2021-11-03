import warnings
import time
import numpy as np
from numpy.core.fromnumeric import argsort
from numpy.core.numeric import indices
import psutil
import os

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
from einops import rearrange

from utils.dataset import split_train_test
from utils.visualize import visualize_face, visualize_faces, visualize_faces_with_row_label, visualize_graph, \
    visualize_confusion_matrix, visualize_faces_with_x_label

warnings.filterwarnings("ignore")


def check_memory(title=None):
    if title is None:
        pid = os.getpid()
        current_process = psutil.Process(pid)
        current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
        return current_process_memory_usage_as_KB
    else:
        print("== " + title)
        # general RAM usage
        memory_usage_dict = dict(psutil.virtual_memory()._asdict())
        memory_usage_percent = memory_usage_dict['percent']
        print(f"memory_usage_percent: {memory_usage_percent}%")
        # current process RAM usage
        pid = os.getpid()
        current_process = psutil.Process(pid)
        current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
        print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")
        print("--" * 30)
        return current_process_memory_usage_as_KB


def recognition_accuracy_pca(dataset):
    x_train = dataset["train_faces"]
    y_train = dataset["train_identities"]
    x_test = dataset["test_faces"]
    y_test = dataset["test_identities"]
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    test_accuracy = np.zeros((364))
    train_accuracy = np.zeros((364))
    x = np.linspace(1, 365, num=365)

    for i_pca in range(1, 364):
        # start_memory = check_memory("Vanilla_Before_Recognition")
        """ 2. PCA """
        pca = PCA(n_components=i_pca)
        x_train_pca = pca.fit(x_train, y_train).transform(x_train)
        x_test_pca = pca.transform(x_test)
        # calculation_memory = check_memory("Vanilla_Before_Recognition")
        clf_pca = KNeighborsClassifier()
        clf_pca.fit(x_train_pca, y_train)
        # recognition_cal_memory = check_memory("Vanilla_Before_Recognition") - start_memory
        # recognition_memory = check_memory("Vanilla_Before_Recognition") - calculation_memory
        train_accuracy[i_pca] = clf_pca.score(x_train_pca, y_train)
        test_accuracy[i_pca] = clf_pca.score(x_test_pca, y_test)
        # print(f"Recognition Memory :  {recognition_memory} KB")
        # print(f"Recognition + Calculation Memory :  {recognition_cal_memory } KB")

    plt.clf()
    plt.plot(np.arange(364), np.array(train_accuracy))
    plt.plot(np.arange(364), np.array(test_accuracy))
    plt.xlabel('M_pca')
    plt.ylabel('Recognition Accuracy')
    plt.legend(['train', 'test'])
    plt.savefig('figure_k_5/PCA Recognition Accuracy')
    plt.show()


def recognition_accuracy_lda(dataset):
    x_train = dataset["train_faces"]
    y_train = dataset["train_identities"]
    x_test = dataset["test_faces"]
    y_test = dataset["test_identities"]
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    test_accuracy = np.zeros((52))
    train_accuracy = np.zeros((52))
    x = np.linspace(1, 53, num=53)

    for i_lda in range(51, 52):
        start_memory = check_memory("Vanilla_Before_Recognition")
        """ 2. LDA """
        lda = LinearDiscriminantAnalysis(n_components=i_lda)
        x_train_lda = lda.fit(x_train, y_train).transform(x_train)
        x_test_lda = lda.transform(x_test)
        calculation_memory = check_memory("Vanilla_Before_Recognition")
        clf_lda = KNeighborsClassifier(n_neighbors=52)
        clf_lda.fit(x_train_lda, y_train)
        recognition_cal_memory = check_memory("Vanilla_Before_Recognition") - start_memory
        recognition_memory = check_memory("Vanilla_Before_Recognition") - calculation_memory
        train_accuracy[i_lda] = clf_lda.score(x_train_lda, y_train)
        test_accuracy[i_lda] = clf_lda.score(x_test_lda, y_test)
        print(f"Recognition Memory :  {recognition_memory} KB")
        print(f"Recognition + Calculation Memory :  {recognition_cal_memory} KB")

    # plt.clf()
    # plt.plot(np.arange(52), np.array(train_accuracy))
    # plt.plot(np.arange(52), np.array(test_accuracy))
    # plt.xlabel('M_lda')
    # plt.ylabel('Recognition Accuracy')
    # plt.legend(['train', 'test'])
    # plt.savefig('figure_k_5/LDA Recognition Accuracy')
    # plt.show()


def recognition_accuracies(dataset):
    x_train = dataset["train_faces"]
    y_train = dataset["train_identities"]
    x_test = dataset["test_faces"]
    y_test = dataset["test_identities"]

    # x_train = dataset["train_faces"][:56]
    # y_train = dataset["train_identities"][:56]
    # x_test = dataset["test_faces"][:14]
    # y_test = dataset["test_identities"][:14]

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    test_accuracy = np.zeros((52, 365))
    train_accuracy = np.zeros((52, 365))
    x = np.linspace(1, 365, num=365)
    y = np.linspace(1, 52, num=52)
    x, y = np.meshgrid(x, y)

    for i_lda in range(1, 52, 1):
        for i_pca in range(i_lda, 365, 1):
            """ 1. PCA """
            # i_pca = 364
            pca = PCA(n_components=i_pca)
            x_train_pca = pca.fit(x_train).transform(x_train)
            x_test_pca = pca.transform(x_test)

            """ 2. LDA """
            lda = LinearDiscriminantAnalysis(n_components=i_lda)
            x_train_lda = lda.fit(x_train_pca, y_train).transform(x_train_pca)
            x_test_lda = lda.transform(x_test_pca)

            """ 3.3 PCA + LDA """
            clf_lda = KNeighborsClassifier()
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
    plt.xlabel('M_pca')
    plt.ylabel('M_lda')
    plt.savefig('Q3/figures/Train_Recognition_Accuracy_.png')
    plt.show()

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, test_accuracy, alpha=0.3, color='blue', edgecolor='blue')
    plt.title('Test')
    plt.xlabel('M_pca')
    plt.ylabel('M_lda')
    plt.savefig('Q3/figures/Test_Recognition_Accuracy_.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.add_argument("--best_m", default=50, type=int, help="Number of best eigen choices")
    # parser.add_argument("--m_pca", default=416 - 52, type=int, help="Dimension in PCA")  # Number of Images - class
    parser.add_argument("--m_pca", default=52 - 1, type=int, help="Dimension in PCA")  # Number of Images - class
    parser.add_argument("--m_lda", default=52 - 1, type=int, help="Dimension in LDA")  # class - 1

    parser.set_defaults(vis=True)

    args = parser.parse_args()
    # face_data = io.loadmat('Q3/data/face.mat')
    face_data = io.loadmat('data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    """ Split Dataset """
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)
    x_train = dataset["train_faces"]
    y_train = dataset["train_identities"]
    x_test = dataset["test_faces"]
    y_test = dataset["test_identities"]

    # x_train = np.flip(x_train, axis=0)
    # y_train = np.flip(y_train, axis=0)
    # x_test = np.flip(x_test, axis=0)
    # y_test = np.flip(y_test, axis=0)
    # recognition_accuracies(dataset=dataset)
    recognition_accuracy_pca(dataset=dataset)
    calculation_time = {}

    """ 0. Normalize """
    x_train_original = x_train.copy()
    x_test_original = x_test.copy()
    average_face = np.average(x_train, axis=0)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    """ 1. PCA """
    start = time.time()
    pca = PCA(n_components=251)
    x_train_pca = pca.fit(x_train).transform(x_train)  # N, Mpca 
    x_test_pca = pca.transform(x_test)
    calculation_time['pca'] = time.time() - start

    subtracted_faces = x_train - average_face
    covariance = (subtracted_faces.T @ subtracted_faces) / subtracted_faces.shape[0]
    scatter_PCA = pca.components_ @ covariance @ pca.components_.T
    scatter_PCA = np.fix(scatter_PCA)
    rank_scatter_PCA = np.linalg.matrix_rank(scatter_PCA)
    rank_scatter_original = np.linalg.matrix_rank(covariance)

    """ 2. LDA """
    start = time.time()
    n_components = 51
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    x_train_lda = lda.fit(x_train_pca, y_train).transform(x_train_pca)
    x_test_lda = lda.transform(x_test_pca)
    # x_train_lda = lda.fit(x_train, y_train).transform(x_train)
    # x_test_lda = lda.transform(x_test)
    calculation_time['lda'] = time.time() - start

    temp_face = rearrange(x_train, '(A B) D -> A B D', A=52, B=8, D=2576)
    temp_mean = np.mean(temp_face, axis=1)
    mean_list = temp_mean - average_face
    scatter_between_temp = 8 * mean_list.T @ mean_list
    original_rank_scatter_between = np.linalg.matrix_rank(scatter_between_temp)
    scatter_Between = pca.components_ @ (scatter_between_temp) @ pca.components_.T
    scatter_Between = np.fix(scatter_Between)
    rank_scatter_Between = np.linalg.matrix_rank(scatter_Between)

    temp_face_within = temp_face - rearrange(temp_mean, 'N D -> N 1 D', N=52, D=2576)
    temp_within = rearrange(temp_face_within, 'A B D -> (A B) D', A=52, B=8, D=2576)
    original_rank_scatter_within = np.linalg.matrix_rank(temp_within)
    scatter_Within = pca.components_ @ (temp_within.T @ temp_within) @ pca.components_.T
    scatter_Within = np.fix(scatter_Within)
    rank_scatter_Within = np.linalg.matrix_rank(scatter_Within)

    scatter_total = np.linalg.inv(scatter_Within) @ scatter_Between
    scatter_total = np.fix(scatter_total)

    scatter_lda = lda.scalings_[:, :n_components].T @ scatter_total @ lda.scalings_[:, :n_components]
    scatter_lda = np.fix(scatter_lda)
    rank_scatter_LDA = np.linalg.matrix_rank(scatter_lda)

    """ 3. NN Classification """
    """ 3.1 Vanilla """
    check_memory("Vanilla_Before_Recognition")
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=52)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    calculation_time['clf_vanilla'] = time.time() - start
    check_memory("Vanilla_After_Recognition")
    print("clf.score             : {0:.3f}".format(clf.score(x_train, y_train)))
    print("clf.score             : {0:.3f}".format(clf.score(x_test, y_test)))

    """ 3.1.1 Examples """
    diff = prediction - rearrange(y_test, 'N 1 -> N', N=104)
    success_case = np.where(diff == 0)[0]
    success_images = np.array(x_test_original[2 * prediction[success_case[2:7]] - 1])

    if args.vis:
        visualize_faces(faces=success_images, n=1, title="success_case_vanilla", cols=5, rows=1)
    fail_case = np.where(diff != 0)[0]
    fail_indices_target = (fail_case // 2) + 1
    fail_images = np.concatenate(
        [x_test_original[2 * fail_indices_target[2:7] - 1], x_test_original[2 * prediction[fail_case[2:7]] - 1]],
        axis=0)

    if args.vis:
        visualize_faces_with_row_label(faces=fail_images, n=1, title="fail_case_vanilla", cols=5, rows=2,
                                       rows_label=['Target', 'Predicted'])

    """ 3.1.2 Confusion Matrix """
    if args.vis:
        visualize_confusion_matrix(y_test, prediction, 'Recognition_Vanilla')

    """ 3.2 PCA """
    check_memory("PCA_Before_Recognition")
    start = time.time()
    clf_pca = KNeighborsClassifier(n_neighbors=52)
    clf_pca.fit(x_train_pca, y_train)
    prediction_pca = clf_pca.predict(x_test_pca)
    calculation_time['clf_pca'] = time.time() - start
    check_memory("PCA_After_Recognition")
    print("clf.score             : {0:.3f}".format(clf_pca.score(x_train_pca, y_train)))
    print("clf.score             : {0:.3f}".format(clf_pca.score(x_test_pca, y_test)))

    """ 3.2.1 Examples """
    diff = prediction_pca - rearrange(y_test, 'N 1 -> N', N=104)
    success_case = np.where(diff == 0)[0]
    success_images = np.array(x_test_original[2 * prediction_pca[success_case[2:7]] - 1])
    if args.vis:
        visualize_faces(faces=success_images, n=1, title="success_case_pca", cols=5, rows=1)
    fail_case = np.where(diff != 0)[0]
    fail_indices_target = (fail_case // 2) + 1
    fail_images = np.concatenate(
        [x_test_original[2 * fail_indices_target[2:7] - 1], x_test_original[2 * prediction_pca[fail_case[2:7]] - 1]],
        axis=0)
    if args.vis:
        visualize_faces_with_row_label(faces=fail_images, n=1, title="fail_case_pca", cols=5, rows=2,
                                       rows_label=['Target', 'Predicted'])

    """ 3.2.2 Confusion Matrix """
    if args.vis:
        visualize_confusion_matrix(y_test, prediction_pca, 'Recognition_PCA')

    """ 3.3 PCA + LDA """
    check_memory("LDA_Before_Recognition")
    start = time.time()
    clf_lda = KNeighborsClassifier()
    clf_lda.fit(x_train_lda, y_train)
    prediction_lda = clf_lda.predict(x_test_lda)
    calculation_time['clf_lda'] = time.time() - start
    check_memory("LDA_After_Recognition")
    print("clf.score             : {0:.3f}".format(clf_lda.score(x_train_lda, y_train)))
    print("clf.score             : {0:.3f}".format(clf_lda.score(x_test_lda, y_test)))

    """ 3.3.1 Examples """
    diff = prediction_lda - rearrange(y_test, 'N 1 -> N', N=104)
    success_case = np.where(diff == 0)[0]
    success_images = np.array(x_test_original[2 * prediction_lda[success_case[2:7]] - 1])
    if args.vis:
        visualize_faces(faces=success_images, n=1, title="success_case_lda", cols=5, rows=1)
    fail_case = np.where(diff != 0)[0]
    fail_indices_target = (fail_case // 2) + 1
    fail_images = np.concatenate(
        [x_test_original[2 * fail_indices_target[2:7] - 1], x_test_original[2 * prediction_lda[fail_case[2:7]] - 1]],
        axis=0)
    if args.vis:
        visualize_faces_with_row_label(faces=fail_images, n=1, title="fail_case_lda", cols=5, rows=2,
                                       rows_label=['Target', 'Predicted'])

    fail_case = np.where(prediction_lda == 4)[0]
    fail_indices_target = (fail_case // 2) + 1
    fail_images = np.concatenate(
        [x_test_original[2 * fail_indices_target[2:7] - 1], x_test_original[2 * prediction_lda[fail_case[2:7]] - 1]],
        axis=0)
    if args.vis:
        visualize_faces_with_row_label(faces=fail_images, n=1, title="fail_case_lda_4", cols=5, rows=2,
                                       rows_label=['Target', 'Predicted'])
        visualize_face(x_test_original[6])  # identity 4
        visualize_faces_with_x_label(np.array(x_test_original[2 * fail_indices_target - 1]), title='predicted_to_ID4',
                                     n=1, cols=6, rows=3, x_label=fail_indices_target)  # identity 4

    """ 3.3.2 Confusion Matrix """
    if args.vis:
        visualize_confusion_matrix(y_test, prediction_lda, 'Recognition_LDA')
    # t = lda.explained_variance_ratio_
    # sort_indices = np.argsort(t)
    eigenvectors = lda.scalings_[:, :n_components]
    # visualize_faces(eigenvectors)
    # .T

    # target = np.array(max_accuracy_target)[indices]
    # target_reconstructed = average_face + (target @ max_accuracy_eigenvectors.T) @ max_accuracy_eigenvectors
    # nearest_neighbor = np.array(max_accuracy_nn)[indices]
    # nearest_neighbor_reconstructed = average_face + (
    #             nearest_neighbor @ max_accuracy_eigenvectors.T) @ max_accuracy_eigenvectors

    # inp = np.concatenate([target, target_reconstructed, nearest_neighbor, nearest_neighbor_reconstructed], axis=0)

    """ 4. Measure Recognition Accuracies """
    # if args.vis:
    # recognition_accuracies(dataset=dataset)
    # recognition_accuracy_lda(dataset=dataset)

    """ 5. Calculation Time """
    for key, value in calculation_time.items():
        print("Calculation Time " + key + ": ", value)

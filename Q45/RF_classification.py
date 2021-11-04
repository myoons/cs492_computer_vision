import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from einops import rearrange
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from SIFT import ready
from utils.visualize import visualize_confusion_matrix, plot_confusion_matrix


def test_n_classifiers(x, x_test, y_true, classifiers, names):
    n_classifiers = len(classifiers)
    n_repeat = 30
    train_scores = np.zeros((n_classifiers, n_repeat))
    test_scores = np.zeros((n_classifiers, n_repeat))
    for i in range(n_classifiers):
        rf_classifier = rf_classifiers[i]
        for j in range(n_repeat):
            rf_classifier.fit(x, y_true)
            y_pred = rf_classifier.predict(x_test)
            train_score = rf_classifier.score(x, y_true)
            test_score = accuracy_score(y_true, y_pred)
            train_scores[i][j] = train_score
            test_scores[i][j] = test_score

    train_score = np.mean(np.array(train_scores), axis=1)
    test_score = np.mean(np.array(test_scores), axis=1)

    for i in range(n_classifiers):
        print("Train Accuracy ({0}): {1:.3f}".format(names[i], train_score[i]))
        print("Test Accuracy ({0}): {1:.3f}".format(names[i], test_score[i]))
        # print(f"Test Accuracy ({names[i]}): {test_score[i]}")

    return 0


def calculate_confusion_matrix(classifier, x, x_test, y_true, save):
    classifier.fit(x, y_true)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y, y_pred)
    if save:
        plot_confusion_matrix(cm, target_names=classes, title='Recognition_Test')
        # visualize_confusion_matrix(cm, title='Recognition_Accuracy_Test')  # Simple Version


def show_success_fail_examples(datasets, y_true, y_pred, save=False):
    # x_test_original = datasets['test_img']
    # diff = y_true - y_pred
    #
    # success_case = np.where(diff == 0)[0]
    # fail_case = np.where(diff != 0)[0]
    # success_indices =
    # fail_indices_target =
    # fail_indices_pred =
    # success_images = np.array(x_test_original[success_indices])
    # fail_images = np.concatenate(
    #     [x_test_original[fail_indices_target], x_test_original[fail_indices_pred]],
    #     axis=0)
    # if save:
        # visualize_faces(faces=success_images, title="success_case", rows=1)
        # visualize_faces_with_row_label(faces=fail_images,
                                       # shape=( , ), title="fail_case", rows_label=['Target', 'Predicted'])
    # else:

    return 0


if __name__ == '__main__':
    plt.ioff()
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.set_defaults(vis=True)
    args = parser.parse_args()

    c = 10
    n_images_per_class = 15
    vocabulary_size = 100
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

    """ Training & Testing Data """
    X = []  # n_images, K
    y = []  # n_images, 1
    for key, value in dataset['train_hist'].items():
        X.append(np.array(value))
        y.append(np.array(CLASS[key]))
    X = rearrange(X, 'c n k -> (c n) k', c=c, n=n_images_per_class, k=vocabulary_size)
    y = np.repeat(y, n_images_per_class)
    X_test = []  # n_images, vocabulary_size
    for key, value in dataset['test_hist'].items():
        X_test.append(np.array(value))
    X_test = rearrange(X_test, 'c n k -> (c n) k', c=c, n=n_images_per_class, k=vocabulary_size)

    """ sample """
    rf_classifier = RandomForestClassifier(n_estimators=256, max_depth=c)  # best_case
    rf_classifier.fit(X, y)
    y_pred = rf_classifier.predict(X_test)
    print("accuracy : ", accuracy_score(y, y_pred))

    """ Experiment of Recognition Accuracies """
    n_estimators = [128, 256, 512]
    max_depth = [None, int(2 * c), c, int(c // 2)]
    max_features = [None, 'auto', 'log2']
    max_features_2 = [1, 2, int(vocabulary_size//10)]
    max_samples = [0.3, 0.6, 0.9, None]
    max_leaf_nodes = [None, c, int(c // 2)]
    # rf_classifier = RandomForestClassifier(n_estimators=n_estimators,
    #                                        max_depth=max_depth,
    #                                        max_features=max_features,
    #                                        max_leaf_nodes=max_leaf_nodes,
    #                                        max_samples=max_samples)

    """ 1. n_estimators """
    # rf_classifiers = [RandomForestClassifier(n_estimators=d) for d in n_estimators]
    # test_n_classifiers(X, X_test, y, rf_classifiers, n_estimators)

    """ 2. max_depth """
    # rf_classifiers = [RandomForestClassifier(n_estimators=256, max_depth=d) for d in max_depth]
    # test_n_classifiers(X, X_test, y, rf_classifiers, max_depth)

    """ 3. max_features """
    print("3. max_features")
    # rf_classifiers = [RandomForestClassifier(n_estimators=256, max_depth=c, max_features=d) for d in max_features]
    # test_n_classifiers(X, X_test, y, rf_classifiers, max_features)
    rf_classifiers = [RandomForestClassifier(n_estimators=256, max_depth=c, max_features=d) for d in max_features_2]
    test_n_classifiers(X, X_test, y, rf_classifiers, max_features_2)

    """ 4. max_samples """
    # rf_classifiers = [RandomForestClassifier(n_estimators=256, max_depth=c, max_samples=d) for d in max_samples]
    # test_n_classifiers(X, X_test, y, rf_classifiers, max_samples)

    """ 5. max_leaf_nodes """
    # rf_classifiers = [RandomForestClassifier(n_estimators=256, max_depth=c, max_leaf_nodes=d) for d in max_leaf_nodes]
    # test_n_classifiers(X, X_test, y, rf_classifiers, max_leaf_nodes)

    """ Confusion Matrix """
    # calculate_confusion_matrix(rf_classifier, X, X_test, y, save=args.vis)

    """ Success / Fail Examples """
    # show_success_fail_examples()


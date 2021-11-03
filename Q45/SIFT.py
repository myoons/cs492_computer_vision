import os
import cv2

import numpy as np
from glob import glob
from joblib import dump, load
from collections import Counter
from sklearn.cluster import KMeans 


def load_images():
    dataset = {'train_img': {}, 'test_img': {}}
    classes = os.listdir(os.path.abspath('Caltech_101'))

    for cls in classes:
        images = glob(f'Caltech_101/{cls}/**.jpg')
        target_images = np.random.choice(images, 30, replace=False)
        dataset['train_img'][cls] = [cv2.imread(img) for img in target_images[:15]]
        dataset['test_img'][cls] = [cv2.imread(img) for img in target_images[15:]]

    return dataset


def images_to_descriptors(images):
    sift = cv2.SIFT.create()

    descriptors = []
    for img in images:
        _, desc = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        descriptors.append(desc)

    return descriptors


def descriptors_to_histogram(descriptors, codebook):
    nearest = codebook.predict(descriptors)
    hist = np.zeros(codebook.n_clusters)

    count = Counter(nearest)
    for idx, n in count.items():
        hist[idx] = n

    hist /= descriptors.shape[0]
    return hist


def train_codebook(descriptors, n_clusters=100, checkpoint=None):
    if checkpoint:
        return load(checkpoint)

    kmeans = KMeans(n_clusters=n_clusters).fit(descriptors)
    dump(kmeans, 'saved_kmeans.pkl')
    return kmeans


""" Constructing dataset (images, descriptors, histogram) & Training CODEBOOK. """
def ready():
    dataset = load_images()
    dataset['train_desc'] = {}
    dataset['test_desc'] = {}

    for (cls, images) in dataset['train_img'].items():
        dataset['train_desc'][cls] = images_to_descriptors(images)
    for (cls, images) in dataset['test_img'].items():
        dataset['test_desc'][cls] = images_to_descriptors(images)

    all_train_descriptors = []
    for descriptors in dataset['train_desc'].values():
        all_train_descriptors.append(np.concatenate(descriptors, axis=0))

    all_train_descriptors = np.concatenate(all_train_descriptors)
    codebook = train_codebook(all_train_descriptors, n_clusters=100, checkpoint='saved_kmeans.pkl')

    dataset['train_hist'] = {}
    dataset['test_hist'] = {}
    for (cls, descriptors) in dataset['train_desc'].items():
        dataset['train_hist'][cls] = np.stack([descriptors_to_histogram(desc, codebook) for desc in descriptors],
                                              axis=0)
    for (cls, descriptors) in dataset['test_desc'].items():
        dataset['test_hist'][cls] = np.stack([descriptors_to_histogram(desc, codebook) for desc in descriptors], axis=0)

    return dataset, codebook


if __name__ == '__main__':
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

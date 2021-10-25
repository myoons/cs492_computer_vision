import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy import io
from numpy import linalg
from argparse import ArgumentParser
from sklearn.decomposition import PCA, IncrementalPCA

from utils.dataset import split_train_test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.add_argument("--n_components", default=50, type=int, help="Dimension of PCA")
    parser.set_defaults(vis=False)

    args = parser.parse_args()

    face_data = io.loadmat('data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    """ Split Dataset """
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)

    """ 1. Batch PCA """
    pca = PCA(n_components=args.n_components)
    batch_pca = pca.fit_transform(dataset["train_faces"])

    """ 2. First Subset PCA """
    incremental_pca = IncrementalPCA(n_components=args.n_components, batch_size=104)
    first_subset_pca = incremental_pca.partial_fit(dataset["train_faces"][:dataset["train_faces"].shape[0] // 4])

    """ 3.  IncrementalPCA  """
    incremental_pca = IncrementalPCA(n_components=args.n_components, batch_size=104)
    incremental_pca.fit_transform(dataset["train_faces"])
import warnings

warnings.filterwarnings("ignore")

import time
import numpy as np
from scipy import io
from tqdm import trange
from numpy import linalg
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.decomposition import PCA, IncrementalPCA

from utils.dataset import split_train_test
from utils.visualize import visualize_faces, visualize_graph


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.set_defaults(vis=False)

    args = parser.parse_args()

    face_data = io.loadmat('data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    """ Split Dataset """
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)

    indices = np.random.choice(len(dataset["train_faces"]), 5, replace=False)
    source = dataset["train_faces"][indices]

    accuracy = defaultdict(list)
    computation_time = defaultdict(list)
    reconstruction_losses = defaultdict(list)
    for n_components in trange(1, 104):
        """ 1. Batch PCA """
        pca = PCA(n_components=n_components)

        start = time.time()
        pca.fit(dataset["train_faces"])
        computation_time['pca'].append(time.time() - start)
        projected_pca = pca.transform(dataset["train_faces"])

        reconstructed_pca = pca.inverse_transform(projected_pca)
        reconstruction_loss = np.average(linalg.norm(reconstructed_pca - dataset["train_faces"], axis=1), axis=0)
        reconstruction_losses['pca'].append(reconstruction_loss)

        correct = 0
        test_projected = pca.transform(dataset["test_faces"])
        for idx in range(test_projected.shape[0]):
            dist = linalg.norm(test_projected[idx] - projected_pca, axis=1)
            min_distance_idx = np.argmin(dist)

            if dataset["test_identities"][idx] == dataset["train_identities"][min_distance_idx]:
                correct += 1
        accuracy['pca'].append(correct / test_projected.shape[0] * 100)

        """ 2. First Subset PCA """
        first_subset = dataset["train_faces"][:104]
        first_pca = IncrementalPCA(n_components=n_components, batch_size=104)

        start = time.time()
        first_pca.partial_fit(first_subset)
        computation_time['first_subset'].append(time.time() - start)
        projected_first_pca = first_pca.transform(dataset["train_faces"])

        reconstructed_first_pca = first_pca.inverse_transform(projected_first_pca)
        reconstruction_loss = np.average(linalg.norm(reconstructed_first_pca - dataset["train_faces"], axis=1), axis=0)
        reconstruction_losses['first_subset'].append(reconstruction_loss)

        correct = 0
        test_projected = first_pca.transform(dataset["test_faces"])
        for idx in range(test_projected.shape[0]):
            dist = linalg.norm(test_projected[idx] - projected_first_pca, axis=1)
            min_distance_idx = np.argmin(dist)

            if dataset["test_identities"][idx] == dataset["train_identities"][min_distance_idx]:
                correct += 1
        accuracy['first_subset'].append(correct / test_projected.shape[0] * 100)

        """ 3.  IncrementalPCA  """
        incremental_pca = IncrementalPCA(n_components=n_components, batch_size=104)
        start = time.time()
        incremental_pca.fit(dataset["train_faces"])
        computation_time['incremental_pca'].append(time.time() - start)
        projected_incremental_pca = incremental_pca.transform(dataset["train_faces"])

        reconstructed_incremental_pca = incremental_pca.inverse_transform(projected_incremental_pca)
        reconstruction_loss = np.average(linalg.norm(reconstructed_incremental_pca - dataset["train_faces"], axis=1), axis=0)
        reconstruction_losses['incremental_pca'].append(reconstruction_loss)

        correct = 0
        test_projected = incremental_pca.transform(dataset["test_faces"])
        for idx in range(test_projected.shape[0]):
            dist = linalg.norm(test_projected[idx] - projected_incremental_pca, axis=1)
            min_distance_idx = np.argmin(dist)

            if dataset["test_identities"][idx] == dataset["train_identities"][min_distance_idx]:
                correct += 1

        accuracy['incremental_pca'].append(correct / test_projected.shape[0] * 100)

        if args.vis:
            reconstructed = reconstructed_pca[indices]
            inp = np.concatenate([source[:5], reconstructed[:5]])
            visualize_faces(inp, n_components=n_components, n=1, title="Train Reconstruction PCA", sub='pca')

            reconstructed = reconstructed_first_pca[indices]
            inp = np.concatenate([source[:5], reconstructed[:5]])
            visualize_faces(inp, n_components=n_components, n=1, title="Train Reconstruction First Subset", sub='first_subset')

            reconstructed = reconstructed_incremental_pca[indices]
            inp = np.concatenate([source[:5], reconstructed[:5]])
            visualize_faces(inp, n_components=n_components, n=1, title="Train Reconstruction Incremental PCA", sub='incremental_pca')

    if args.vis:

        visualize_graph(x_axis=np.arange(1, 104),
                        y_axes=[accuracy['pca'], accuracy['first_subset'], accuracy['incremental_pca']],
                        xlabel="M",
                        ylabel="Identity Recognition Accuracy",
                        legend=['PCA', 'First Subset', 'Incremental PCA'],
                        title="Identity Recognition Accuracy")

        visualize_graph(x_axis=np.arange(1, 104),
                        y_axes=[computation_time['pca'], computation_time['first_subset'], computation_time['incremental_pca']],
                        xlabel="M",
                        ylabel="Computation time",
                        legend=['PCA', 'First Subset', 'Incremental PCA'],
                        title="Computation time")

        visualize_graph(x_axis=np.arange(1, 104),
                        y_axes=[reconstruction_losses['pca'], reconstruction_losses['first_subset'], reconstruction_losses['incremental_pca']],
                        xlabel="M",
                        ylabel="Reconstruction Loss",
                        legend=['PCA', 'First Subset', 'Incremental PCA'],
                        title="Reconstruction Loss")

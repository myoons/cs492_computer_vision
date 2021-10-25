import warnings

warnings.filterwarnings("ignore")

import time
import numpy as np
from scipy import io
from tqdm import trange
from numpy import linalg
from argparse import ArgumentParser

from utils.dataset import split_train_test
from utils.visualize import visualize_face, visualize_faces, visualize_graph, visualize_tsne

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.add_argument("--best_m", default=50, type=int, help="Number of best eigen choices")
    parser.set_defaults(vis=False)

    args = parser.parse_args()

    face_data = io.loadmat('data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    """ Split Dataset """
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)

    if args.vis:
        visualize_faces(dataset['test_faces'], identities=dataset['test_identities'], n=1, random=False,
                        title="Dataset")

    average_face = np.average(dataset['train_faces'], axis=0)
    if args.vis:
        visualize_face(average_face, title="Average Face")

    # Subtract average face
    subtracted_faces = dataset['train_faces'] - average_face

    """ Compute Eigenface """
    start = time.time()
    covariance = (subtracted_faces.T @ subtracted_faces) / subtracted_faces.shape[0]

    eigenvalues, eigenvectors = linalg.eig(covariance)
    print(f"Original Eigenface Calculation Time : {time.time() - start:.3f}")

    eigenvalues, eigenvectors = eigenvalues.astype(float), np.swapaxes(eigenvectors.astype(float), 0, 1)
    sort_indices = np.argsort(eigenvalues)[::-1]
    best_m_eigenvectors = eigenvectors[sort_indices[:args.best_m]]

    if args.vis:
        visualize_faces(best_m_eigenvectors.astype(float), n=1, random=False, title="Best Eigenvectors")

    """ Compute Low Computation Eigenface """
    start = time.time()
    low_covariance = (subtracted_faces @ subtracted_faces.T) / subtracted_faces.shape[0]

    low_eigenvalues, low_eigenvectors = linalg.eig(low_covariance)
    proj = low_eigenvectors.T @ subtracted_faces
    low_eigenvectors = proj / linalg.norm(proj, axis=1)[..., np.newaxis]
    print(f"Low Eigenface Calculation Time : {time.time() - start:.3f}")

    low_eigenvalues, low_eigenvectors = low_eigenvalues.astype(float), low_eigenvectors.astype(float)
    low_sort_indices = np.argsort(low_eigenvalues)[::-1]
    low_best_m_eigenvectors = low_eigenvectors[low_sort_indices[:args.best_m]]

    if args.vis:
        visualize_faces(low_best_m_eigenvectors.astype(float), n=1, random=False, title="Best Low Eigenvectors")

    """ Visualize Reconstruction """
    if args.vis:
        reconstruct_indices = np.random.choice(dataset["train_faces"].shape[0], 5, replace=False)
        target = subtracted_faces[reconstruct_indices]

        reconstructed = average_face + (target @ best_m_eigenvectors.T) @ best_m_eigenvectors
        visualize_faces(np.concatenate([target, reconstructed], axis=0), n=1,
                        title="Train Face Reconstruction with Eigenvectors")

        reconstructed = average_face + (target @ low_best_m_eigenvectors.T) @ low_best_m_eigenvectors
        visualize_faces(np.concatenate([target, reconstructed], axis=0), n=1,
                        title="Train Face Reconstruction with Low Eigenvectors")

    """ Train Reconstruction Loss """
    train_reconstruction_losses = []
    for m in trange(1, dataset["train_faces"].shape[0] + 1):
        temp_eigenvectors = eigenvectors[sort_indices[:m]]
        reconstructed = average_face + (subtracted_faces @ temp_eigenvectors.T) @ temp_eigenvectors
        train_reconstruction_losses.append(
            np.average(linalg.norm(reconstructed - dataset["train_faces"], axis=1), axis=0))

    train_low_reconstruction_losses = []
    for m in trange(1, dataset["train_faces"].shape[0] + 1):
        temp_eigenvectors = low_eigenvectors[low_sort_indices[:m]]
        reconstructed = average_face + (subtracted_faces @ temp_eigenvectors.T) @ temp_eigenvectors
        train_low_reconstruction_losses.append(
            np.average(linalg.norm(reconstructed - dataset["train_faces"], axis=1), axis=0))

    """ Test Reconstruction Loss """
    test_subtracted_faces = dataset["test_faces"] - average_face

    test_reconstruction_losses = []
    for m in trange(1, dataset["test_faces"].shape[0] + 1):
        temp_eigenvectors = eigenvectors[sort_indices[:m]]
        reconstructed = average_face + (test_subtracted_faces @ temp_eigenvectors.T) @ temp_eigenvectors
        test_reconstruction_losses.append(
            np.average(linalg.norm(reconstructed - dataset["test_faces"], axis=1), axis=0))

    test_low_reconstruction_losses = []
    for m in trange(1, dataset["test_faces"].shape[0] + 1):
        temp_eigenvectors = low_eigenvectors[low_sort_indices[:m]]
        reconstructed = average_face + (test_subtracted_faces @ temp_eigenvectors.T) @ temp_eigenvectors
        test_low_reconstruction_losses.append(
            np.average(linalg.norm(reconstructed - dataset["test_faces"], axis=1), axis=0))

    """ Visualize Reconstruction Losses """
    if args.vis:
        visualize_graph(x_axis=np.arange(1, dataset["train_faces"].shape[0] + 1),
                        y_axes=[train_reconstruction_losses, train_low_reconstruction_losses],
                        xlabel="M",
                        ylabel="Reconstruction Train Loss",
                        legend=['Original', 'Low Computation'],
                        title="Reconstruction Train Loss")

        visualize_graph(x_axis=np.arange(1, dataset["test_faces"].shape[0] + 1),
                        y_axes=[test_reconstruction_losses, test_low_reconstruction_losses],
                        xlabel="M",
                        ylabel="Reconstruction Test Loss",
                        legend=['Original', 'Low Computation'],
                        title="Reconstruction Test Loss")

        visualize_graph(x_axis=np.arange(1, dataset["test_faces"].shape[0] + 1),
                        y_axes=[train_reconstruction_losses[:dataset["test_faces"].shape[0]],
                                train_low_reconstruction_losses[:dataset["test_faces"].shape[0]],
                                test_reconstruction_losses,
                                test_low_reconstruction_losses],
                        xlabel="M",
                        ylabel="Reconstruction Loss",
                        legend=['Original_Train', 'Low Computation_Train', 'Original_Test', 'Low Computation_Test'],
                        title="Reconstruction Loss")

    """ t-SNE """
    if args.vis:
        n_identities = 5

        visualize_tsne(data=subtracted_faces[:n_identities * 8],
                       identities=dataset["train_identities"][:n_identities * 8],
                       title="t-SNE Train Faces")

        visualize_tsne(data=subtracted_faces[:n_identities * 8] @ best_m_eigenvectors.T,
                       identities=dataset["train_identities"][:n_identities * 8],
                       title="t-SNE Train Projected")

        visualize_tsne(data=test_subtracted_faces[:n_identities * 2] @ best_m_eigenvectors.T,
                       identities=dataset["test_identities"][:n_identities * 2],
                       title="t-SNE Test Projected")

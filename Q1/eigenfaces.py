import warnings


import time
import numpy as np
from scipy import io
from tqdm import trange
from numpy import linalg
from argparse import ArgumentParser

from utils.dataset import split_train_test
from utils.visualize import visualize_face, visualize_faces, visualize_graph, visualize_tsne, visualize_3d
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.add_argument("--best_m", default=50, type=int, help="Number of best eigen choices")
    parser.set_defaults(vis=False)

    args = parser.parse_args()

    face_data = io.loadmat('Q1/data/face.mat')
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
    eigenvectors = eigenvectors[sort_indices]
    best_m_eigenvectors = eigenvectors[:args.best_m]

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
    low_eigenvectors = low_eigenvectors[low_sort_indices]
    low_best_m_eigenvectors = low_eigenvectors[:args.best_m]

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
    for m in trange(1, dataset["train_faces"].shape[0]):
        m_eigenvectors = eigenvectors[:m]
        reconstructed = average_face + (subtracted_faces @ m_eigenvectors.T)  @ m_eigenvectors
        train_reconstruction_losses.append(np.average(linalg.norm(reconstructed - dataset["train_faces"], axis=1), axis=0))

    train_low_reconstruction_losses = []
    for m in trange(1, dataset["train_faces"].shape[0]):
        m_eigenvectors = low_eigenvectors[:m]
        reconstructed = average_face + (subtracted_faces @ m_eigenvectors.T)  @ m_eigenvectors
        train_low_reconstruction_losses.append(np.average(linalg.norm(reconstructed - dataset["train_faces"], axis=1), axis=0))

    """ Test Reconstruction Loss """
    test_subtracted_faces = dataset["test_faces"] - average_face

    test_reconstruction_losses = []
    for m in trange(1, dataset["train_faces"].shape[0]):
        m_eigenvectors = eigenvectors[:m]
        reconstructed = average_face + (test_subtracted_faces @ m_eigenvectors.T)  @ m_eigenvectors
        test_reconstruction_losses.append(np.average(linalg.norm(reconstructed - dataset["test_faces"], axis=1), axis=0))

    test_low_reconstruction_losses = []
    for m in trange(1, dataset["train_faces"].shape[0]):
        m_eigenvectors = low_eigenvectors[:m]
        reconstructed = average_face + (test_subtracted_faces @ m_eigenvectors.T)  @ m_eigenvectors
        test_low_reconstruction_losses.append(np.average(linalg.norm(reconstructed - dataset["test_faces"], axis=1), axis=0))

    """ Visualize Reconstruction Losses """
    if args.vis:
        visualize_graph(x_axis=np.arange(1, dataset["train_faces"].shape[0]),
                        y_axes=[train_reconstruction_losses, train_low_reconstruction_losses],
                        xlabel="M",
                        ylabel="Reconstruction Train Loss",
                        legend=['Original', 'Low Computation'],
                        title="Reconstruction Train Loss")

        visualize_graph(x_axis=np.arange(1, dataset["train_faces"].shape[0]),
                        y_axes=[test_reconstruction_losses, test_low_reconstruction_losses],
                        xlabel="M",
                        ylabel="Reconstruction Test Loss",
                        legend=['Original', 'Low Computation'],
                        title="Reconstruction Test Loss")

        visualize_graph(x_axis=np.arange(1, dataset["train_faces"].shape[0]),
                        y_axes=[train_reconstruction_losses,
                                train_low_reconstruction_losses,
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

    """ Nearest Neighbor """
    list_accuracy = []
    max_accuracy = 0
    max_accuracy_eigenvectors = None
    max_accuracy_target, max_accuracy_nn = [], []
    for m in trange(1, dataset["train_faces"].shape[0]):
        m_eigenvectors = low_eigenvectors[low_sort_indices[:m]]
        train_projected = subtracted_faces @ m_eigenvectors.T
        test_projected = test_subtracted_faces @ m_eigenvectors.T

        correct = 0
        target, predict = [], []
        for idx in range(test_projected.shape[0]):
            dist = linalg.norm(test_projected[idx] - train_projected, axis=1)
            min_distance_idx = np.argmin(dist)

            if dataset["test_identities"][idx] == dataset["train_identities"][min_distance_idx]:
                correct += 1
            else:
                target.append(dataset["test_faces"][idx])
                predict.append(dataset["train_faces"][min_distance_idx])

        accuracy = correct / test_projected.shape[0]
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_accuracy_target = target
            max_accuracy_nn = predict
            max_accuracy_eigenvectors = m_eigenvectors

        list_accuracy.append(accuracy)

    if args.vis:
        visualize_graph(x_axis=np.arange(1, dataset["train_faces"].shape[0]),
                        y_axes=[list_accuracy],
                        xlabel="M",
                        ylabel="Identity Recognition Test Accuracy",
                        legend=['Test'],
                        title="Identity Recognition (NN) Test Accuracy")

    if args.vis:
        assert len(max_accuracy_target) == len(max_accuracy_nn)
        indices = np.random.choice(len(max_accuracy_target), 5, replace=False)

        target = np.array(max_accuracy_target)[indices]
        target_reconstructed = average_face + (target @ max_accuracy_eigenvectors.T) @ max_accuracy_eigenvectors
        nearest_neighbor = np.array(max_accuracy_nn)[indices]
        nearest_neighbor_reconstructed = average_face + (
                    nearest_neighbor @ max_accuracy_eigenvectors.T) @ max_accuracy_eigenvectors

        inp = np.concatenate([target, target_reconstructed, nearest_neighbor, nearest_neighbor_reconstructed], axis=0)
        visualize_faces(inp, n=1, rows=4, cols=5, title="Nearest Neighbor Fail Cases")

    """ 3-Dimension Projection (M=3) """
    m_eigenvectors = low_eigenvectors[:3]
    train_projected = subtracted_faces @ m_eigenvectors.T
    test_projected = test_subtracted_faces @ m_eigenvectors.T

    if args.vis:
        n_identities = 5

        visualize_3d(projections=train_projected[:n_identities * 8],
                     identities=dataset["train_identities"][:n_identities * 8],
                     title="Train Faces 3D projected (M=3)")

        visualize_3d(projections=test_projected[:n_identities * 2],
                     identities=dataset["test_identities"][:n_identities * 2],
                     title="Test Faces 3D projected (M=3)")

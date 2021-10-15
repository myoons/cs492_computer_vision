import numpy as np
from scipy import io
from numpy import linalg
from argparse import ArgumentParser

from utils.dataset import split_train_test
from utils.visualize import visualize_face, visualize_faces


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.add_argument("--best_m", default=50, type=int, help="Number of best eigen choices")
    parser.set_defaults(vis=False)

    args = parser.parse_args()

    # Load face dataset
    face_data = io.loadmat('data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    # Partition the provided face data into train & test
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)

    # Visualize data
    if args.vis:
        visualize_faces(dataset['test_faces'], identities=dataset['test_identities'], n=1, random=False)

    # Compute average face vector and visualize
    average_face = np.average(dataset['train_faces'], axis=1)
    if args.vis:
        visualize_face(average_face)

    # Subtract average face
    subtracted_faces = dataset['train_faces'] - np.repeat(average_face[..., np.newaxis], dataset['train_faces'].shape[1], axis=1)

    # Compute covariance matrix S
    covariance = (subtracted_faces @ subtracted_faces.T) / subtracted_faces.shape[1]

    # Compute the eigenvectors of covariance
    eigenvalues, eigenvectors = linalg.eig(covariance)
    eigenvalues, eigenvectors = eigenvalues.astype(np.float), eigenvectors.astype(np.float)
    best_m_eigenvalues, best_m_eigenvectors = eigenvalues[:args.best_m], eigenvectors[..., :args.best_m]

    if args.vis:
        visualize_faces(best_m_eigenvectors.astype(np.float), n=1, random=False)

    # Reconstruct face with eigenfaces
    reconstruct_indices = np.random.choice(dataset["train_faces"].shape[1], 5, replace=False)

    target_faces, reconstructed_faces = [], []
    for idx in reconstruct_indices:
        target_face = dataset["train_faces"][..., idx]
        reconstructed = average_face + np.sum(best_m_eigenvectors * (target_face @ best_m_eigenvectors), axis=1)

        target_faces.append(target_face)
        reconstructed_faces.append(reconstructed)

    if args.vis:
        visualize_faces(np.swapaxes(np.array(target_faces + reconstructed_faces), 0, 1), n=1)

    # Use low-dimensional computation of eigenspace
    low_covariance = (subtracted_faces.T @ subtracted_faces) / subtracted_faces.shape[1]

    # Compute the eigenvectors of covariance
    low_eigenvalues, low_eigenvectors = linalg.eig(low_covariance)
    low_eigenvectors = (subtracted_faces @ low_eigenvectors) / linalg.norm(subtracted_faces @ low_eigenvectors, axis=0)
    low_eigenvalues, low_eigenvectors = low_eigenvalues.astype(np.float), low_eigenvectors.astype(np.float)

    low_best_m_eigenvalues, low_best_m_eigenvectors = low_eigenvalues[:args.best_m], low_eigenvectors[..., :args.best_m]

    if args.vis:
        visualize_faces(low_best_m_eigenvectors.astype(np.float), n=1, random=False)

    # Reconstruct face with eigenfaces
    reconstruct_indices = np.random.choice(dataset["train_faces"].shape[1], 5, replace=False)

    target_faces, reconstructed_faces = [], []
    for idx in reconstruct_indices:
        target_face = dataset["train_faces"][..., idx]
        reconstructed = average_face + np.sum(best_m_eigenvectors * (target_face @ best_m_eigenvectors), axis=1)

        target_faces.append(target_face)
        reconstructed_faces.append(reconstructed)

    if args.vis:
        visualize_faces(np.swapaxes(np.array(target_faces + reconstructed_faces), 0, 1), n=1)

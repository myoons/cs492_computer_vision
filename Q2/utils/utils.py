import time
import numpy as np
from tqdm import trange
from numpy import linalg
from scipy.linalg import orth


def reconstruct(dataset, eigenvectors):
    subtracted_faces = dataset['train_subtracted_faces']
    reconstructed = dataset['average_face'] + (subtracted_faces @ eigenvectors.T) @ eigenvectors
    return reconstructed


def batch_pca(train_faces):
    mean_face = np.average(train_faces, axis=0)
    subtracted_faces = train_faces - mean_face

    covariance = (subtracted_faces.T @ subtracted_faces) / subtracted_faces.shape[0]
    low_covariance = (subtracted_faces @ subtracted_faces.T) / subtracted_faces.shape[0]
    start = time.time()
    eigenvalues, eigenvectors = linalg.eig(low_covariance)
    end = time.time()
    eigenvectors = eigenvectors.T @ subtracted_faces / linalg.norm(eigenvectors.T @ subtracted_faces, axis=1)[
        ..., np.newaxis]
    eigenvalues, eigenvectors = eigenvalues.astype(float), eigenvectors.astype(float)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[sort_indices]

    return covariance, eigenvalues, eigenvectors, mean_face, end - start


def merge_pca(dataset, split=52):
    n = split // 52
    all_indices = np.arange(dataset['train_faces'].shape[0])

    f_train_faces = dataset['train_faces'][all_indices % 8 < n]
    s_train_faces = dataset['train_faces'][all_indices % 8 >= n]

    f_cov, f_w, f_v, f_m, f_t = batch_pca(f_train_faces)
    s_cov, s_w, s_v, s_m, s_t = batch_pca(s_train_faces)

    n_f, n_s = f_train_faces.shape[0], s_train_faces.shape[0]
    n_t = n_f + n_s
    scatter = (n_f * n_s / (n_t ** 2)) * ((f_m - s_m) @ (f_m - s_m).T)
    t_cov = (n_f / n_t) * f_cov + (n_s / n_t) * s_cov + scatter
    phi = orth(np.concatenate([f_v[:len(s_v) // 2], s_v[:len(s_v) // 2], (f_m - s_m)[np.newaxis, ...]]).T).T
    reduced = phi @ t_cov @ phi.T

    third_start = time.time()
    t_w, t_v = linalg.eig(reduced)
    third_end = time.time()

    t_v = t_v.T @ phi / linalg.norm(t_v.T @ phi, axis=1)[..., np.newaxis]
    t_w, t_v = t_w.astype(float), t_v.astype(float)
    sort_indices = np.argsort(t_w)[::-1]
    t_v = t_v[sort_indices]

    t_t = third_end - third_start

    joint_t = f_t + s_t + t_t
    incremental_t = s_t + t_t
    return t_w, t_v, joint_t, incremental_t


def incremental_pca(dataset, partition=104):
    assert dataset['train_faces'].shape[0] % partition == 0

    times = []
    n = dataset['train_faces'].shape[0] // partition
    all_indices = np.arange(dataset['train_faces'].shape[0])
    trained_train_faces = dataset['train_faces'][all_indices % n == 0]
    n_trained = trained_train_faces.shape[0]
    trained_cov, _, trained_v, trained_m, t = batch_pca(trained_train_faces)
    times.append(t)

    for i in range(1, n):
        target_train_faces = dataset['train_faces'][all_indices % n == i]
        n_target = target_train_faces.shape[0]
        target_cov, _, target_v, target_m, t = batch_pca(target_train_faces)
        times.append(t)

        n_trained, trained_m, trained_cov, trained_v, t = merge_two_models(n_trained, n_target, trained_m, target_m,
                                                                           trained_cov, target_cov, trained_v, target_v)

        times.append(t)

    assert n_trained == dataset['train_faces'].shape[0]

    spent_time = sum(times)
    return trained_v, spent_time


def merge_two_models(n_f, n_s, f_m, s_m, f_cov, s_cov, f_v, s_v):
    n_t = n_f + n_s
    t_m = (n_f * f_m + n_s * s_m) / n_t

    scatter = (n_f * n_s / (n_t ** 2)) * ((f_m - s_m) @ (f_m - s_m).T)
    t_cov = (n_f / n_t) * f_cov + (n_s / n_t) * s_cov + scatter
    phi = orth(np.concatenate([f_v[:len(s_v) // 2], s_v[:len(s_v) // 2], (f_m - s_m)[np.newaxis, ...]]).T).T
    reduced = phi @ t_cov @ phi.T

    third_start = time.time()
    t_w, t_v = linalg.eig(reduced)
    third_end = time.time()

    t_v = t_v.T @ phi / linalg.norm(t_v.T @ phi, axis=1)[..., np.newaxis]
    t_w, t_v = t_w.astype(float), t_v.astype(float)
    sort_indices = np.argsort(t_w)[::-1]
    t_v = t_v[sort_indices]

    t_t = third_end - third_start
    return n_t, t_m, t_cov, t_v, t_t


def calculate_reconstruction_loss(dataset, eigenvectors, test=False, m=50):
    if test:
        subtracted_faces = dataset["train_subtracted_faces"]
    else:
        subtracted_faces = dataset["test_subtracted_faces"]

    list_reconstruction_loss = []
    for m in trange(1, m + 1):
        m_eigenvectors = eigenvectors[:m]
        reconstructed = dataset["average_face"] + (subtracted_faces @ m_eigenvectors.T) @ m_eigenvectors
        list_reconstruction_loss.append(np.average(linalg.norm(reconstructed - dataset["test_faces"], axis=1), axis=0))

    return list_reconstruction_loss


def calculate_accuracy(dataset, eigenvectors, m=50):
    list_accuracy = []
    for m in trange(1, m + 1):
        m_eigenvectors = eigenvectors[:m]
        train_projected = dataset["train_subtracted_faces"] @ m_eigenvectors.T
        test_projected = dataset["test_subtracted_faces"] @ m_eigenvectors.T

        dist = linalg.norm(np.repeat(test_projected[:, np.newaxis, :], 416, axis=1) - train_projected, axis=2)
        min_distance_idx = np.argmin(dist, axis=1)

        correct = dataset["test_identities"] == dataset["train_identities"][min_distance_idx]
        accuracy = np.sum(correct) / dataset["n_test"] * 100
        list_accuracy.append(accuracy)

    return list_accuracy

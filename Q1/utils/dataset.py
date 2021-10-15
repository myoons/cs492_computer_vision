import numpy as np


def split_train_test(faces, identities, r=0.8, n_per_identity=10):
    assert faces.shape[1] == identities.shape[1], print("length of faces and identities are different")

    split = int(n_per_identity * r)
    all_indices = np.arange(faces.shape[1])

    dataset = dict()
    dataset['train_faces'] = faces[..., (all_indices % n_per_identity) < split]
    dataset['train_identities'] = identities[..., (all_indices % n_per_identity) < split]
    dataset['test_faces'] = faces[..., (all_indices % n_per_identity) >= split]
    dataset['test_identities'] = identities[..., (all_indices % n_per_identity) >= split]

    return dataset

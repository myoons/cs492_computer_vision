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

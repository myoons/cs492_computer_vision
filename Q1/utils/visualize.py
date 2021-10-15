import numpy as np
import matplotlib.pyplot as plt


def visualize_face(face, shape=(46, 56)):
    face = np.swapaxes(np.reshape(face, shape), 0, 1)
    plt.imshow(face, cmap='gray')
    plt.show()


def visualize_faces(faces, n=1, shape=(46, 56), random=False, identities=None):
    if identities is not None:
        assert faces.shape[1] == identities.shape[1], print("length of faces and identities are different")

    cols = 5
    rows = 2
    for i in range(n):

        axes = []
        fig = plt.figure(figsize=(20, 10), dpi=100)

        if faces.shape[1] < 10:
            indices = np.arange(faces.shape[1])
        else:
            if random:
                indices = np.random.choice(faces.shape[1], 10, replace=True)
            else:
                indices = 10 * i + np.arange(10)

        for pos, idx in enumerate(indices):
            face = np.swapaxes(np.reshape(faces[..., idx], shape), 0, 1)
            axes.append(fig.add_subplot(rows, cols, pos+1))

            if identities is not None:
                identity = str(identities[..., idx])
                axes[-1].set_title(identity)

            plt.imshow(face, cmap='gray')

        plt.show()

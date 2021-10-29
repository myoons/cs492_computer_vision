import os

import numpy as np
import matplotlib.pyplot as plt


def visualize_faces(faces, n=1, shape=(46, 56), random=False, identities=None, title=None, cols=5, rows=2, sub=None):
    if identities is not None:
        assert faces.shape[0] == identities.shape[0], print("length of faces and identities are different")

    for i in range(n):

        axes = []
        fig = plt.figure(figsize=(20, 10), dpi=100)
        if title is not None:
            fig.suptitle(title, fontsize=20)

        n = rows * cols
        if faces.shape[0] < n:
            indices = np.arange(faces.shape[0])
        else:
            if random:
                indices = np.random.choice(faces.shape[0], n, replace=False)
            else:
                indices = n * i + np.arange(n)

        for pos, idx in enumerate(indices):
            face = np.swapaxes(np.reshape(faces[idx], shape), 0, 1)
            axes.append(fig.add_subplot(rows, cols, pos+1))

            if identities is not None:
                identity = str(identities[idx])
                axes[-1].set_title(identity)

            plt.imshow(face, cmap='gray')

        os.makedirs(f'figures/{sub}', exist_ok=True)

        if title:
            plt.savefig(f"figures/{sub}/{title.replace(' ', '_').lower()}.png")
        else:
            plt.show()

    plt.close()


def visualize_graph(x_axis, y_axes, xlabel, ylabel, legend, title=None):
    for y in y_axes:
        plt.plot(x_axis, y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)

    if title is not None:
        plt.title(title)
        plt.savefig(f"figures/{title.replace(' ', '_').lower()}.png")
    else:
        plt.show()

    plt.close()
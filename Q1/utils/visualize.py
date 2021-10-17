import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_face(face, shape=(46, 56), title=None):
    face = np.swapaxes(np.reshape(face, shape), 0, 1)

    if title is not None:
        plt.title(title)
        plt.savefig(f"figures/{title.replace(' ', '_').lower()}.png")
    else:
        plt.imshow(face, cmap='gray')
        plt.show()

    plt.close()


def visualize_faces(faces, n=1, shape=(46, 56), random=False, identities=None, title=None):
    if identities is not None:
        assert faces.shape[0] == identities.shape[0], print("length of faces and identities are different")

    cols = 5
    rows = 2
    for i in range(n):

        axes = []
        fig = plt.figure(figsize=(20, 10), dpi=100)
        if title is not None:
            fig.suptitle(title, fontsize=20)

        if faces.shape[0] < 10:
            indices = np.arange(faces.shape[0])
        else:
            if random:
                indices = np.random.choice(faces.shape[0], 10, replace=False)
            else:
                indices = 10 * i + np.arange(10)

        for pos, idx in enumerate(indices):
            face = np.swapaxes(np.reshape(faces[idx], shape), 0, 1)
            axes.append(fig.add_subplot(rows, cols, pos+1))

            if identities is not None:
                identity = str(identities[idx])
                axes[-1].set_title(identity)

            plt.imshow(face, cmap='gray')

        if title:
            plt.savefig(f"figures/{title.replace(' ', '_').lower()}.png")
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


def visualize_tsne(data, identities, title=None):
    assert data.shape[0] == identities.shape[0]

    tsne = TSNE(n_components=3, n_iter=5000)
    tsne_results = tsne.fit_transform(X=data)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2],
                 c=identities,
                 alpha=0.8)

    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')

    if title is not None:
        plt.title(title)
        plt.savefig(f"figures/{title.replace(' ', '_').lower()}.png")
    else:
        plt.show()

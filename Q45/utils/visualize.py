import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def visualize_image(face, shape=(46, 56), title=None):
    face = np.swapaxes(np.reshape(face, shape), 0, 1)
    plt.imshow(face, cmap='gray')

    if title is not None:
        plt.title(title)
        plt.savefig(f"figures/{title.replace(' ', '_').lower()}.png")
    else:
        plt.show()

    plt.close()


def visualize_images(faces, n=1, shape=(46, 56), random=False, identities=None, title=None, cols=5, rows=2):
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
            axes.append(fig.add_subplot(rows, cols, pos + 1))

            if identities is not None:
                identity = str(identities[idx])
                axes[-1].set_title(identity)

            plt.imshow(face, cmap='gray')

        if title:
            plt.savefig(f"figures/{title.replace(' ', '_').lower()}.png")
        else:
            plt.show()

    plt.close()


def visualize_faces_with_row_label(faces, n=1, shape=(46, 56), random=False, identities=None, title=None, cols=5,
                                   rows=2, rows_label=None):
    if rows_label is not None:
        assert (len(rows_label) == rows)

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

        row_idx = 0
        for pos, idx in enumerate(indices):
            face = np.swapaxes(np.reshape(faces[idx], shape), 0, 1)
            ax = fig.add_subplot(rows, cols, pos + 1)

            if idx % cols == 0:
                ax.set_ylabel(rows_label[row_idx])
                row_idx += 1

            axes.append(ax)

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

    plt.close()


def visualize_3d(projections, identities, title=None):
    assert projections.shape[0] == identities.shape[0]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(projections[:, 0], projections[:, 1], projections[:, 2],
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

    plt.close()


def visualize_confusion_matrix(cm, title):
    # cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('Target')

    if title is not None:
        plt.title(title)
        plt.savefig(f"figures/{title.replace(' ', '_').lower()}.png")
    else:
        plt.show()


def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        # plt.xticks(tick_marks, target_names)  # overlap
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.tight_layout()
    plt.title(title)
    if title is not None:
        plt.savefig(f"figures/{title.replace(' ', '_').lower()}.png")
    else:
        plt.show()



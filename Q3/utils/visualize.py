from matplotlib import pyplot as plt
from scipy import io
from sklearn.metrics import confusion_matrix
import numpy as np

def visualize_confusion_matrix(y_test, prediction, title):
    cm = confusion_matrix(y_test, prediction)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.title(title)
    plt.savefig("Q3/figures/" + title + ".png")
    # plt.show()

def visualize_face(face, shape=(46, 56), title=None):
    face = np.swapaxes(np.reshape(face, shape), 0, 1)
    plt.imshow(face, cmap='gray')

    if title is not None:
        plt.title(title)
        plt.savefig("Q3/figures/" + title + ".png")
    else:
        plt.show()

    plt.close()


def visualize_faces(faces, n=1, shape=(46, 56), random=False, identities=None, title=None, cols=5, rows=2):
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

        if title:
          plt.savefig("Q3/figures/" + title + ".png")
        else:
            plt.show()

    plt.close()


def visualize_faces_with_row_label(faces, n=1, shape=(46, 56), random=False, identities=None, title=None, cols=5, rows=2):
    if identities is not None:
        assert faces.shape[0] == identities.shape[0], print("length of faces and identities are different")
    rows_label = ['target_image', 'target_neighbor', 'predicted_image', 'nearest_neighbor']

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
            ax = fig.add_subplot(rows, cols, pos+1)
            
            if(idx % cols == 0):
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

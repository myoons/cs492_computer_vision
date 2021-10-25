from matplotlib import pyplot as plt
from scipy import io
from sklearn.metrics import confusion_matrix


def visualize_confusion_matrix(y_test, prediction, title):
    cm = confusion_matrix(y_test, prediction)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.title(title)
    plt.savefig("Q3/figures/" + title + ".png")
    # plt.show()

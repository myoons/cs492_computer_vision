import warnings

warnings.filterwarnings("ignore")

from scipy import io
from argparse import ArgumentParser

from utils.utils import *
from utils.dataset import split_train_test
from utils.visualize import visualize_faces, visualize_graph

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", dest="vis", action="store_true", help="Visualize images")
    parser.set_defaults(vis=False)

    args = parser.parse_args()

    face_data = io.loadmat('data/face.mat')
    faces, identities = face_data['X'], face_data['l']

    """ Split Dataset """
    dataset = split_train_test(faces, identities, r=0.8, n_per_identity=10)

    dataset['average_face'] = np.average(dataset['train_faces'], axis=0)
    dataset['train_subtracted_faces'] = dataset['train_faces'] - dataset['average_face']
    dataset['test_subtracted_faces'] = dataset['test_faces'] - dataset['average_face']
    all_indices = np.arange(dataset['train_faces'].shape[0])

    M = 50
    N_SPLIT = 104
    N_FIRST_SUBSET = 104

    """ Number of images in first model 에 따라서 실험을 바꿔야 한다. """
    _, _, batch_v, _, batch_t = batch_pca(dataset['train_faces'])
    batch_reconstruction_loss = calculate_reconstruction_loss(dataset, batch_v, m=M)
    batch_accuracy = calculate_accuracy(dataset, batch_v, m=M)

    # _, _, fs_v, _, fs_t = batch_pca(dataset['train_faces'][all_indices % 4 == 0])
    # fs_reconstruction_loss = calculate_reconstruction_loss(dataset, fs_v, m=M)
    # fs_accuracy = calculate_accuracy(dataset, fs_v, m=M)
    #
    # inc_52_v, inc_52_t = incremental_pca(dataset, 52)
    # inc_52_reconstruction_loss = calculate_reconstruction_loss(dataset, inc_52_v, m=M)
    # inc_52_accuracy = calculate_accuracy(dataset, inc_52_v, m=M)
    #
    # inc_104_v, inc_104_t = incremental_pca(dataset, 104)
    # inc_104_reconstruction_loss = calculate_reconstruction_loss(dataset, inc_104_v, m=M)
    # inc_104_accuracy = calculate_accuracy(dataset, inc_104_v, m=M)
    #
    # inc_208_v, inc_208_t = incremental_pca(dataset, 208)
    # inc_208_reconstruction_loss = calculate_reconstruction_loss(dataset, inc_208_v, m=M)
    # inc_208_accuracy = calculate_accuracy(dataset, inc_208_v, m=M)

    # visualize_faces(batch_v, title=f"Batch PCA", sub="eigenface")
    # visualize_faces(fs_v, title=f"First Subset PCA", sub="eigenface")
    # visualize_faces(inc_104_v, title=f"Incremental PCA", sub="eigenface")

    # visualize_graph(x_axis=np.arange(1, M + 1),
    #                 y_axes=[batch_accuracy, fs_accuracy, inc_accuracy],
    #                 xlabel="M",
    #                 ylabel="Accuracy",
    #                 legend=['PCA', 'First Subset PCA', 'Incremental PCA'],
    #                 title=f"Accuracy")
    #
    # visualize_graph(x_axis=np.arange(1, M + 1),
    #                 y_axes=[batch_reconstruction_loss, fs_reconstruction_loss, inc_reconstruction_loss],
    #                 xlabel="M",
    #                 ylabel="Reconstruction Loss",
    #                 legend=['PCA', 'First Subset PCA', 'Incremental PCA'],
    #                 title=f"Reconstruction Loss")

    # visualize_graph(x_axis=np.arange(1, M + 1),
    #                 y_axes=[batch_accuracy, inc_52_accuracy, inc_104_accuracy, inc_208_accuracy],
    #                 xlabel="M",
    #                 ylabel="Accuracy",
    #                 legend=['Batch', 'Inc 52', 'Inc 104', 'Inc 208'],
    #                 title=f"Accuracy")
    #
    # visualize_graph(x_axis=np.arange(1, M + 1),
    #                 y_axes=[batch_reconstruction_loss, inc_52_reconstruction_loss, inc_104_reconstruction_loss, inc_208_reconstruction_loss],
    #                 xlabel="M",
    #                 ylabel="Reconstruction Loss",
    #                 legend=['Batch', 'Inc 52', 'Inc 104', 'Inc 208'],
    #                 title=f"Reconstruction Loss")

    """ Split """
    list_accuracy, list_reconstruction_loss = [batch_accuracy[-1]], [batch_reconstruction_loss[-1]]
    list_joint, list_incremental, list_batch = [batch_t], [batch_t], [batch_t]

    for split in range(52, 416, 52):
        merge_w, merge_v, joint_t, incremental_t = merge_pca(dataset, split=split)
        list_joint.append(joint_t)
        list_incremental.append(incremental_t)
        list_batch.append(batch_t)

        # merge_reconstruction_loss = calculate_reconstruction_loss(dataset, merge_v, m=M)
        # merge_accuracy = calculate_accuracy(dataset, merge_v, m=M)

        # list_accuracy.append(merge_accuracy[-1])
        # list_reconstruction_loss.append(merge_reconstruction_loss[-1])

    list_batch.append(batch_t)
    list_joint.append(batch_t)
    list_incremental.append(0.)
    list_accuracy.append(batch_accuracy[-1])
    list_reconstruction_loss.append(batch_reconstruction_loss[-1])

    list_accuracy_b, list_reconstruction_loss_b = [batch_accuracy[-1]] * len(list_accuracy), [batch_reconstruction_loss[-1]] * len(list_reconstruction_loss)

    if args.vis:
        visualize_graph(x_axis=np.arange(0, 417, 52),
                        y_axes=[list_batch, list_joint, list_incremental],
                        xlabel="Number of images in first model",
                        ylabel="computation time (s)",
                        legend=['batch time', 'joint time', 'incremental time'],
                        title="Computation Time")

        # visualize_graph(x_axis=np.arange(0, 417, 52),
        #                 y_axes=[list_reconstruction_loss_b, list_reconstruction_loss],
        #                 xlabel="Number of images in first model",
        #                 ylabel=f"Reconstruction Loss (M={M})",
        #                 legend=['batch', 'incremental'],
        #                 title="Reconstruction Loss")
        #
        # visualize_graph(x_axis=np.arange(0, 417, 52),
        #                 y_axes=[list_accuracy_b, list_accuracy],
        #                 xlabel="Number of images in first model",
        #                 ylabel=f"Accuracy (M={M}) ",
        #                 legend=['batch', 'incremental'],
        #                 title="Accuracy")

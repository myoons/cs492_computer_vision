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

    N_FIRST_SUBSET = 104
    M = 80

    """ Number of images in first model 에 따라서 실험을 바꿔야 한다. """
    _, _, batch_v, _, batch_t = batch_pca(dataset['train_faces'])
    batch_reconstruction_loss = calculate_reconstruction_loss(dataset, batch_v, m=M)
    batch_accuracy = calculate_accuracy(dataset, batch_v, m=M)

    _, _, fs_v, _, fs_t = batch_pca(dataset['train_faces'][:N_FIRST_SUBSET])
    fs_reconstruction_loss = calculate_reconstruction_loss(dataset, fs_v, m=M)
    fs_accuracy = calculate_accuracy(dataset, fs_v, m=M)

    inc_v, inc_t = incremental_pca(dataset, 208)
    inc_reconstruction_loss = calculate_reconstruction_loss(dataset, inc_v, m=M)
    inc_accuracy = calculate_accuracy(dataset, inc_v, m=M)

    """ Split """
    # list_accuracy, list_reconstruction_loss = [np.max(batch_accuracy)], [np.min(batch_reconstruction_loss)]
    # list_joint, list_incremental, list_batch = [batch_t], [batch_t], [batch_t]

    # for split in range(52, 416, 52):
    #     merge_w, merge_v, joint_t, incremental_t = merge_pca(dataset, split=split)
    #     list_joint.append(joint_t)
    #     list_incremental.append(incremental_t)
    #     list_batch.append(batch_t)
    #
    #     merge_reconstruction_loss = calculate_reconstruction_loss(dataset, merge_v, m=M)
    #     merge_accuracy = calculate_accuracy(dataset, merge_v, m=M)
    #
    #     list_accuracy.append(np.max(merge_accuracy))
    #     list_reconstruction_loss.append(np.min(merge_reconstruction_loss))
    #
    # list_batch.append(batch_t)
    # list_joint.append(batch_t)
    # list_incremental.append(0.)
    # list_accuracy.append(np.max(batch_accuracy))
    # list_reconstruction_loss.append(np.min(batch_reconstruction_loss))
    #
    # list_accuracy_b, list_reconstruction_loss_b = [np.max(batch_accuracy)] * len(list_accuracy), [np.min(batch_reconstruction_loss)] * len(list_reconstruction_loss)

    # if args.vis:
    #     visualize_graph(x_axis=np.arange(0, 417, 52),
    #                     y_axes=[list_batch, list_joint, list_incremental],
    #                     xlabel="Number of images in first model",
    #                     ylabel="computation time (s)",
    #                     legend=['batch time', 'joint time', 'incremental time'],
    #                     title="Incremental Computation Time")
    #
    #     visualize_graph(x_axis=np.arange(0, 417, 52),
    #                     y_axes=[list_reconstruction_loss_b, list_reconstruction_loss],
    #                     xlabel="Number of images in first model",
    #                     ylabel=f"Reconstruction Loss (M={M})",
    #                     legend=['batch', 'incremental'],
    #                     title="Incremental Reconstruction Loss")
    #
    #     visualize_graph(x_axis=np.arange(0, 417, 52),
    #                     y_axes=[list_accuracy_b, list_accuracy],
    #                     xlabel="Number of images in first model",
    #                     ylabel=f"Accuracy (M={M}) ",
    #                     legend=['batch', 'incremental'],
    #                     title="Incremental Accuracy")

    print(
        f"""
    Batch PCA : {batch_t:.5f} seconds
    First Subset PCA : {fs_t:.5f} seconds
    Incremental PCA : {inc_t:.5f} seconds
        """)

    visualize_faces(batch_v, title="batch", sub="eigenface")
    visualize_faces(fs_v, title="first_subset", sub="eigenface")
    visualize_faces(inc_v, title="incremental", sub="eigenface")

    visualize_graph(x_axis=np.arange(1, M + 1),
                    y_axes=[batch_accuracy, fs_accuracy, inc_accuracy],
                    xlabel="M",
                    ylabel="Accuracy",
                    legend=['PCA', 'First Subset PCA', 'Incremental PCA'],
                    title="Accuracy")

    visualize_graph(x_axis=np.arange(1, M + 1),
                    y_axes=[batch_reconstruction_loss, fs_reconstruction_loss, inc_reconstruction_loss],
                    xlabel="M",
                    ylabel="Reconstruction Loss",
                    legend=['PCA', 'First Subset PCA', 'Incremental PCA'],
                    title="Reconstruction Loss")

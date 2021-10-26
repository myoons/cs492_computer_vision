from __future__ import print_function
import sandbox
import time
import numpy as np
from einops import rearrange, repeat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io


def show_image(images, h, w, title):
    plot_n = h * w
    fig = plt.figure(figsize=(6, 6))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(plot_n):
        ax = fig.add_subplot(h, w, i + 1, xticks=[], yticks=[])
        ax.imshow(images[i].T, cmap=plt.cm.bone, interpolation=None)
    plt.savefig(title)
    plt.show()


def calculate_reconstruction_error(x_gt, x_mean, eigen_vectors, rank):
    H = 46
    W = 56
    Reconstruction_Error = []
    for M in range(rank):
        indices = np.arange(M)
        principal_eigen_vectors = np.take(eigen_vectors, indices, axis=-1)  # D, M
        proj_x = (x_gt - x_mean) @ principal_eigen_vectors  # N_test, M
        proj_x_inverse = proj_x @ (principal_eigen_vectors.T)  # N_test, D
        result = proj_x_inverse + x_mean  # N_test, D
        diff = np.linalg.norm((x_gt - result), ord=2, axis=1)
        error = np.mean(diff)
        Reconstruction_Error.append(error)
        # Qualitative Result
        # ============================================================
        # plot_n = 4
        # images = np.take(result, np.arange(plot_n), axis=0)
        # images_gt = np.take(x_gt, np.arange(plot_n), axis=0)
        # images = rearrange(images, 'N (H W) -> N H W', N=plot_n, H=H, W=W)
        # images_gt = rearrange(images_gt, 'N (H W) -> N H W', N=plot_n, H=H, W=W)
        # if(M % 100 == 0):
        # show_image(np.concatenate([images_gt, images], axis=0), 2, plot_n, )
        # ============================================================
    return Reconstruction_Error


def qualitative_comparison_of_Reconstruction(x_train, x_test, x_mean, eig, eig_LC):
    H = 46
    W = 56
    rank = (int)(400 / 50)
    modes = [x_train, x_test]
    for mode in range(2):
        x_gt = modes[mode]
        for m in range(1, rank + 1):
            M = m * 50
            indices = np.arange(M)
            principal_eigen_vectors = np.take(eig, indices, axis=-1)  # D, M
            proj_x = (x_gt - x_mean) @ principal_eigen_vectors  # N_test, M
            proj_x_inverse = proj_x @ (principal_eigen_vectors.T)  # N_test, D
            result = proj_x_inverse + x_mean  # N_test, D

            principal_eigen_vectors_LC = np.take(eig_LC, indices, axis=-1)  # D, M
            proj_x_LC = (x_gt - x_mean) @ principal_eigen_vectors_LC  # N_test, M
            proj_x_inverse_LC = proj_x_LC @ (principal_eigen_vectors_LC.T)  # N_test, D
            result_LC = proj_x_inverse_LC + x_mean  # N_test, D
            # Qualitative Result
            # ============================================================
            plot_n = 4
            images = np.take(result, np.arange(plot_n), axis=0)
            images_LC = np.take(result_LC, np.arange(plot_n), axis=0)
            images_gt = np.take(x_gt, np.arange(plot_n), axis=0)

            images = rearrange(images, 'N (H W) -> N H W', N=plot_n, H=H, W=W)
            images_LC = rearrange(images_LC, 'N (H W) -> N H W', N=plot_n, H=H, W=W)
            images_gt = rearrange(images_gt, 'N (H W) -> N H W', N=plot_n, H=H, W=W)

            w = 3
            h = plot_n
            fig = plt.figure(figsize=(9, 9))
            rows = ['ID {}'.format(row) for row in range(1, plot_n + 1)]
            for i in range(w * h):
                ax = fig.add_subplot(h, w, i + 1, xticks=[], yticks=[])
                if (i == 0):
                    ax.set_title('ground truth')
                elif (i == 1):
                    ax.set_title('PCA')
                elif (i == 2):
                    ax.set_title('Low Computation PCA')

                j = i % 3
                k = (int)(i / 3)
                if (j == 0):
                    ax.set_ylabel(rows[k])
                    ax.imshow(images_gt[k].T, cmap=plt.cm.bone, interpolation=None)
                elif (j == 1):
                    ax.imshow(images[k].T, cmap=plt.cm.bone, interpolation=None)
                else:
                    ax.imshow(images_LC[k].T, cmap=plt.cm.bone, interpolation=None)

            if mode == 0:
                model = 'train'
            else:
                model = 'test'
            fig.suptitle(model + '\n(PCs = {})'.format(M))
            plt.savefig('Q1/figures/' + model + '_{}'.format(M))
            # plt.show()
            # ============================================================

    return;


def main():
    # Data
    mat = scipy.io.loadmat('Q1/face.mat')
    H = 46
    W = 56
    D = H * W
    n_image = len(mat['l'][0])
    N = n_image
    n_image_test = int(n_image * (0.2))
    n_image_train = n_image - n_image_test
    n_image_per_person = 10
    n_people = int(n_image / n_image_per_person)
    images_data = np.array(mat['X'])
    images_data = np.transpose(images_data)
    labels = np.array(mat['l'])
    labels = np.transpose(labels)
    x_train, x_test, y_train, y_test = train_test_split(images_data, labels,
                                                        test_size=0.2, shuffle=True, stratify=labels, random_state=34)
    x_mean = np.mean(x_train, axis=0)  # 1, D
    x = x_train - x_mean  # N, D
    x = x.T  # D, N

    time_original = []
    time_LC = []
    for i in range(30):
        # Training Method 1
        start = time.time()
        cov_x = (x @ x.T) / N  # D, D
        eig_values, eig_vectors = np.linalg.eig(cov_x, )
        eig_values = np.real(eig_values)
        eig_vectors = np.real(eig_vectors)
        idx = eig_values.argsort()[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]
        # print("time (Original PCA):", time.time() - start)
        time_original.append(time.time() - start)

        # Training Method 2 - low computation
        start = time.time()
        cov_x_LC = (x.T @ x) / N
        eig_values_LC, eig_vectors_LC = np.linalg.eig(cov_x_LC, )
        eig_values_LC = np.real(eig_values_LC)
        eig_vectors_LC = np.real(eig_vectors_LC)
        idx = eig_values_LC.argsort()[::-1]
        eig_values_LC = eig_values_LC[idx]
        eig_vectors_LC = eig_vectors_LC[:, idx]
        eig_vectors_LC = x @ eig_vectors_LC
        eig_vectors_LC = eig_vectors_LC / np.linalg.norm(eig_vectors_LC, axis=0)
        # print("time (Low Computation):", time.time() - start)
        time_LC.append(time.time() - start)
    print("time (Original PCA):", np.mean(np.array(time_original)))
    print("time (Low Computation):", np.mean(np.array(time_LC)))


    # Difference of Two Methods in EigenVector, EigenValue
    # ====================================================================
    original_eig_vectors = np.take(eig_vectors, np.arange(416), axis=-1)
    diff_of_eigvalue = eig_values[:416] - eig_values_LC
    diff_of_eigvalue_bool = np.where(diff_of_eigvalue < 10 ** -6, True, False)
    diff_of_eigvectors = original_eig_vectors - eig_vectors_LC
    diff_of_eigvectors_bool = np.where((diff_of_eigvectors < 10 ** -6), True, False)

    zero_eig_idx = np.where(eig_values < 10 ** -6)
    zero_eig_idx_LC = np.where(eig_values_LC < 10 ** -6)
    rank_of_eig = int(zero_eig_idx[0][0]);
    rank_of_eig_LC = int(zero_eig_idx_LC[0][0]);
    # ====================================================================

    # Plot Mean Face
    # ====================================================================
    images = rearrange(x_mean, '(H W) -> 1 H W', H=H, W=W)
    show_image(images, 1, 1, 'Q1/figures/mean_face')
    # ====================================================================

    # Plot Eigen Vectors
    # ====================================================================
    temp_eig_vectors = rearrange(eig_vectors, '(H W) N  -> N H W', N=H * W, H=H, W=W)
    show_image(temp_eig_vectors, 2, 5, 'Q1/figures/eigen_Faces')
    # ====================================================================
    # Q. 여기서 x_mean을 더해야 하나?

    # Reconstruction
    # ====================================================================
    # Reconstruction_Error_Train = calculate_reconstruction_error(
    #   x_gt=x_train, x_mean=x_mean, eigen_vectors=eig_vectors, rank=rank_of_eig)

    # Reconstruction_Error_Test = calculate_reconstruction_error(
    #   x_gt=x_test, x_mean=x_mean, eigen_vectors=eig_vectors, rank=rank_of_eig)

    # Reconstruction_Error_Train_LC = calculate_reconstruction_error(
    #   x_gt=x_train, x_mean=x_mean, eigen_vectors=eig_vectors_LC, rank=rank_of_eig_LC)

    # Reconstruction_Error_Test_LC = calculate_reconstruction_error(
    #   x_gt=x_test, x_mean=x_mean, eigen_vectors=eig_vectors_LC, rank=rank_of_eig_LC)
    # plt.clf()
    # plt.plot(np.arange(rank_of_eig), np.array(Reconstruction_Error_Train))
    # plt.plot(np.arange(rank_of_eig), np.array(Reconstruction_Error_Test))
    # plt.plot(np.arange(rank_of_eig_LC), np.array(Reconstruction_Error_Train_LC))
    # plt.plot(np.arange(rank_of_eig_LC), np.array(Reconstruction_Error_Test_LC))
    # plt.xlabel('Principal Component')
    # plt.ylabel('Reconstruction Loss')
    # plt.legend(['train', 'test', 'train(LC)', 'test(LC)'])
    # plt.savefig('Q1/figures/Reconstruction Loss')
    # plt.show()
    # ====================================================================

    # Reconstruction for Qualitative Comparison
    # qualitative_comparison_of_Reconstruction(x_train, x_test, x_mean, eig_vectors, eig_vectors_LC)


if __name__ == "__main__":
    main()

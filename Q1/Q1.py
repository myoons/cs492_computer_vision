# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from __future__ import print_function
from IPython import get_ipython
# %matplotlib inline

# get_ipython().system('pip install -U ipykernel')



import argparse
import os
import time

from numpy.lib.arraysetops import setdiff1d
from numpy.linalg.linalg import eig
from torch._C import dtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data 
import numpy as np
from IPython.display import HTML
from einops import rearrange, repeat
import sklearn.cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score
import scipy.io


def GaussianMatrix(X,sigma):
    row,col=X.shape
    GassMatrix=np.zeros(shape=(row,row))
    X=np.asarray(X)
    i=0
    for v_i in X:
        j=0
        for v_j in X:
            GassMatrix[i,j]=Gaussian(v_i.T,v_j.T,sigma)
            j+=1
        i+=1
    return GassMatrix

def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))

def sandbox():
  ground_truth = [0,1,0]

  X = GaussianMatrix(np.array([[1,2],[3,1],[1,1]]) , 1)
  print("Gaussian distance :\n ", X)
  D = np.sum(X, axis=0)
  L = torch.diag_embed(torch.tensor(D)) - X
  print("\nLaplacian : \n", L)
  w, v = np.linalg.eig(L)
  print("\neigen value: \n", w.real)
  print("\neigen vector:\n", v.real)

  kmeans = sklearn.cluster.KMeans(2).fit(v.real)
  print("\nresult:\n", kmeans.labels_)

  NMI = normalized_mutual_info_score(ground_truth, kmeans.labels_)
  print("\n NMI:\n", NMI)

  A = torch.tensor([[1,1,-1],[1,1,-1],[-1,-1,2]],dtype=float)
  print(A)
  w, v = np.linalg.eig(A)
  u, s, vh = np.linalg.svd(A, full_matrices=False)

  print(w.real)
  print( v.real)
  print(u, s, vh)

  v = torch.tensor([[0.33, 0.59, 0.74],  [0.59, -0.74, 0.33],[-0.74, -0.33, 0.59]])
  w = torch.diag_embed( torch.tensor([11.34, 0.17, -0.52]))
  ans = torch.transpose(v,0,1)@w@v
  print(ans)
  v, w = np.linalg.eig(ans)
  print(v.real)
  print(w.real)

  X = np.array([[1, 2,3], [3,2,1], [5,7,1],
                [2, 3, 5], [1,7, 4], [1,1,1]])
  clustering = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(X)
  clustering.labels_

def show_image(images, h, w):
  plot_n = h * w
  fig = plt.figure(figsize=(6,6))
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
  for i in range(plot_n):
    ax = fig.add_subplot(h, w, i+1, xticks=[], yticks=[])
    ax.imshow(images[i].T, cmap=plt.cm.bone, interpolation='nearest')
  plt.show()

def calculate_reconstruction_error(x_gt, x_mean, eigen_vectors, rank):
  Reconstruction_Error = []
  for M in range(rank):
    indices = np.arange(M)
    principal_eigen_vectors = np.take(eigen_vectors, indices, axis=-1) # D, M 
    proj_x = (x_gt - x_mean) @ principal_eigen_vectors # N_test, M
    proj_x_inverse = proj_x @ (principal_eigen_vectors.T) # N_test, D
    result = proj_x_inverse + x_mean # N_test, D
    diff = np.linalg.norm((x_gt - result), ord=2, axis=1)
    error = np.mean(diff) 
    Reconstruction_Error.append(error)
    # Qualitative Result
    # ============================================================
    # plot_n = 10
    # images = np.take(result, np.arange(plot_n), axis=0)
    # images_gt = np.take(x_gt, np.arange(plot_n), axis=0)
    # images = rearrange(images, 'N (H W) -> N H W', N=plot_n, H=H, W=W)
    # images_gt = rearrange(images_gt, 'N (H W) -> N H W', N=plot_n, H=H, W=W)
    # if(M % 100 == 0):
    #   show_image(np.concatenate([images_gt, images], axis=0), 2, plot_n)
    # ============================================================
  return Reconstruction_Error

def main():
  # Data
  mat = scipy.io.loadmat('./face.mat')
  H = 46
  W = 56
  D = H*W
  n_image = len(mat['l'][0])
  N = n_image 
  n_image_test = int( n_image*(0.2))
  n_image_train = n_image - n_image_test
  n_image_per_person = 10
  n_people = int(n_image / n_image_per_person)
  images_data = np.array(mat['X'])
  images_data = np.transpose(images_data)
  labels = np.array(mat['l'])
  labels = np.transpose(labels)
  x_train, x_test, y_train, y_test = train_test_split(images_data, labels,
            test_size=0.2, shuffle=True, stratify=labels, random_state = 34)

  # Training
  x_mean = np.mean(x_train, axis=0) # 1, D
  x = x_train - x_mean # N, D
  x = x.T # D, N

  # Method 1
  start = time.time() 
  cov_x = (x@x.T) / N # D, D
  eig_values, eig_vectors = np.linalg.eig(cov_x,)
  eig_values = np.real(eig_values)
  eig_vectors = np.real(eig_vectors)
  idx = eig_values.argsort()[::-1]   
  eig_values = eig_values[idx]
  eig_vectors = eig_vectors[:,idx]
  print("time 1:", time.time() - start) 
  
  # Method 2 - low computation
  start = time.time() 
  cov_x_LC = (x.T @ x) / N
  eig_values_LC, eig_vectors_LC = np.linalg.eig(cov_x_LC,)
  eig_values_LC = np.real(eig_values_LC)
  eig_vectors_LC = np.real(eig_vectors_LC)
  idx = eig_values_LC.argsort()[::-1]   
  eig_values_LC = eig_values_LC[idx]
  eig_vectors_LC = eig_vectors_LC[:,idx]
  print("time 2:", time.time() - start) 
  

  # Difference of Two Methods in EigenVector, EigenValue
  # ====================================================================
  original_eig_vectors = np.take( eig_vectors, np.arange(416), axis=-1)
  estimated_eig_vectors = x @ eig_vectors_LC
  estimated_eig_vectors = estimated_eig_vectors / np.linalg.norm(estimated_eig_vectors, axis=0)
  diff_of_eigvalue = eig_values[:416] - eig_values_LC
  diff_of_eigvalue_bool = np.where(diff_of_eigvalue < 10**-6, True, False)
  diff_of_eigvectors = original_eig_vectors - estimated_eig_vectors
  diff_of_eigvectors_bool = np.where((diff_of_eigvectors < 10**-6), True, False)

  zero_eig_idx = np.where(eig_values < 10**-6)
  zero_eig_idx_LC = np.where(eig_values_LC < 10**-6)
  rank_of_eig = int(zero_eig_idx[0][0]);
  rank_of_eig_LC = int(zero_eig_idx_LC[0][0]);
  # ====================================================================
  

  # Plot Mean Face  
  # ====================================================================
  images = rearrange(x_mean, '(H W) -> 1 H W', H=H, W=W)
  show_image(images, 1, 1)
  # ====================================================================

  # Plot Eigen Vectors  
  # ====================================================================
  # temp_eig_vectors = rearrange(eig_vectors, '(H W) N  -> N H W', N=H*W, H=H, W=W)
  # show_image(temp_eig_vectors, 2, 5)
  # ====================================================================
  # Q. 여기서 x_mean을 더해야 하나?


  # Reconstruction
  # ====================================================================
  Reconstruction_Error_Train = calculate_reconstruction_error(
    x_gt=x_train, x_mean=x_mean, eigen_vectors=eig_vectors, rank=rank_of_eig)
  
  Reconstruction_Error_Test = calculate_reconstruction_error(
    x_gt=x_test, x_mean=x_mean, eigen_vectors=eig_vectors, rank=rank_of_eig)
  
  Reconstruction_Error_Train_LC = calculate_reconstruction_error(
    x_gt=x_train, x_mean=x_mean, eigen_vectors=estimated_eig_vectors, rank=rank_of_eig_LC)
  
  Reconstruction_Error_Test_LC = calculate_reconstruction_error(
    x_gt=x_test, x_mean=x_mean, eigen_vectors=estimated_eig_vectors, rank=rank_of_eig_LC)
  
  plt.plot(np.arange(rank_of_eig), np.array(Reconstruction_Error_Train))
  plt.plot(np.arange(rank_of_eig), np.array(Reconstruction_Error_Test))
  plt.plot(np.arange(rank_of_eig_LC), np.array(Reconstruction_Error_Train_LC))
  plt.plot(np.arange(rank_of_eig_LC), np.array(Reconstruction_Error_Test_LC))
  plt.xlabel('Principal Component')
  plt.ylabel('Reconstruction Loss')
  plt.legend(['train', 'test', 'train(LC)', 'test(LC)'])
  plt.show()
  # ====================================================================


if __name__ == "__main__":
  	main()
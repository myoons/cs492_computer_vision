# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from __future__ import print_function
from IPython import get_ipython

# get_ipython().system('pip install -U ipykernel')



#%matplotlib inline
import argparse
import os
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

def main():
  # Spectral Clustering
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

  v1 = (0.2+0.3+0.4+0.5+0.1) / 3.0
  v2 = 0.2+0.3+0.1
  v3 = (0.5+0.1+0.1+0.4)/2.0

  mat = scipy.io.loadmat('./face.mat')
  print("face data", mat['X'], mat['l']);


  
if __name__ == "__main__":
  	main()
  
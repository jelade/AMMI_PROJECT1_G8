import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Get Data: Do not touch it.
def get_data():
  data_url = "http://lib.stat.cmu.edu/datasets/boston"
  raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
  X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
  y = raw_df.values[1::2, 2]
  return X,y

X,y  = get_data()

X.shape

# cgs
def cgs(A):
  """
    Q,R = cgs(A)
    Apply classical Gram-Schmidt to mxn rectangular/square matrix. 

    Parameters
    -------
    A: mxn rectangular/square matrix   

    Returns
    -------
    Q: mxn square matrix
    R: nxn upper triangular matrix

  """
  m = len(A)
  n = len(A[0])
  R = np.zeros((n,n))
  Q = np.ones((m,n))
  for k in range(n):
    w = A[:,k]
    for j in range(1,k-1):
      R[j,k] = np.dot(Q[:,j],w)
    for j in range(k):
        w = w - R[j,k]*Q[:,j]
    R[k,k] = np.linalg.norm(w)
    Q[:,k] = w/R[k,k]

  return R,Q
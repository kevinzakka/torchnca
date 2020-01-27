import time

import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from ipdb import set_trace


class NCA:
  """Neighbourhood Components Analysis.
  """
  def __init__(self, dim=None, init="random"):
    """Constructor.

    Args:
      dim (int): The dimension of the the learned
        linear map A. If no dimension is provided,
        we assume a square matrix A of same dimension
        as the input data. For small values of dim
        (i.e. 2, 3), NCA will do dimensionality reduction.
      init (str): The type of initialization to use for
        the matrix A.
          - `random`: A = N(O, I)
          - `identity`: A = I
          - `whitening`: A = Σ^(-0.5)
    """
    self.dim = dim
    self.init = init

  def __call__(self, X):
    """Apply the learned linear map to the input.
    """
    return X @ self.A.T

  def _init_transformation(self):
    """Initialize the linear transformation A.
    """
    if self.dim is None:
      self.dim = self.num_dims
    if self.init == "random":
      self.A = np.random.randn(self.dim, self.num_dims)
    elif self.init == "identity":
      self.A = np.eye(self.dim, self.num_dims)
    elif self.init == "whitening":
      pass
    else:
      raise ValueError("[!] {} initialization is not supported.".format(init))

  def _softmax(self, x):
    """Numerically stable softmax implementation.
    """
    np.fill_diagonal(x, -np.inf)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1)

  def _objective_func(self, A, X, y, y_mask):
    N, D = X.shape
    A = A.reshape(-1, D)

    # compute pairwise squared Euclidean distances
    # in transformed space
    distances = squareform(pdist(X @ A.T, 'sqeuclidean'))

    # compute pairwise probability matrix p_ij
    # defined by a softmax over negative squared
    # distances in the transformed space.
    # since we are dealing with negative values
    # with the largest value being 0, we need
    # not worry about numerical instabilities
    # in the softmax function
    p_ij = self._softmax(-distances)

    # for each p_i, zero out any p_ij that
    # is not of the same class label as i.
    p_ij_mask = p_ij * y_mask

    # sum over js to compute p_i
    p_i = np.sum(p_ij_mask, axis=1)

    # compute expected number of points
    # correctly classified by summing
    # over all p_i's
    p_total = np.sum(p_i)

    # to maximize the above expectation
    # we negate it and minimize it
    loss = -p_total

    # compute the gradient of the cost function
    # with respect to A
    outer_sum1 = np.zeros((D, D))
    outer_sum2 = np.zeros((D, D))
    for i in range(N):
      diff = X[i] - X
      p = p_ij[i]
      outer_sum1 += (p_i[i] * np.einsum('i,ij,ik->jk', p, diff, diff))
      p = p_ij_mask[i]
      outer_sum2 += np.einsum('i,ij,ik->jk', p, diff, diff)
    grad = 2 * A @ (outer_sum1 - outer_sum2)

    return loss, grad.ravel()

  def train(self, X, y):
    """Trains NCA until convergence.

    Specifically, we maximize the expected number of points
    correctly classified under a stochastic selection rule.
    This rule is defined using a softmax over Euclidean distances
    in the transformed space.

    Args:
      X (ndarray): The dataset of shape (N, D) where
        `D` is the dimension of the feature space and `N`
        is the number of training examples.
      y (ndarray): The class labels of shape (N,).
    """
    self.num_train, self.num_dims = X.shape

    # initialize the linear transformation matrix A
    self._init_transformation()

    # compute pairwise boolean class matrix
    y_mask = y[:, None] == y[None, :]

    ret = minimize(
      self._objective_func,
      self.A,  # initial solution
      args=(X, y, y_mask),
      method="CG",
      jac=True,
      options={
        'disp': True,
        'maxiter': 50,
      },
    )

    self.A = ret['x'].reshape(-1, self.num_dims)
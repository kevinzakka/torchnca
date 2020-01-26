import numpy as np

from scipy.spatial.distance import pdist, squareform
from ipdb import set_trace


class NCA:
  """Neighbourhood Components Analysis.
  """
  def __init__(self, X, y, dim=None, init="identity"):
    """Constructor.

    Args:
      X (ndarray): The dataset of shape (D, N) where
        `D` is the dimension of the feature space and `N`
        is the number of training examples.
      dim (int): The dimension of the the learned
        linear map A. If no dimension is provided,
        we assume a square matrix A of same dimension
        as the input data. For small values of dim
        (i.e. 2, 3), NCA will do dimensionality reduction.
      init (str): The type of initialization to use for
        the matrix A.
          - `identity`: A = I
          - `whitening`: A = Σ^(-0.5)
    """
    D, N = X.shape
    if dim is None:
      dim = D
    if init == "identity":
      self.A = np.eye(dim, D)
    elif init == "whitening":
      pass
    else:
      raise ValueError("[!] {} initialization is not supported.".format(init))

    # store the data
    self.X = X
    self.y = y

  def _softmax(self, x):
    """Numerically stable softmax implementation.
    """
    maxes = np.max(x, axis=1)
    exp = np.exp(x - maxes)
    return exp / np.sum(exp, axis=1)

  def _objective_func(self, A, X, y):
    # apply linear transformation
    prod = A @ X

    # compute pairwise Euclidean distances
    distances = squareform(pdist(prod.T, 'euclidean'))

    # compute probas defined by softmax over distances
    probas = self._softmax(distances)
    np.fill_diagonal(probas, 0)

    set_trace()

  def _grad(self, A):
    pass

  def train(self, max_iters=10000):
    """Trains NCA until convergence.

    Specifically, we maximize the expected number of points
    correctly classified under a stochastic selection rule.
    This rule is defined using a softmax over Euclidean distances
    in the transformed space.
    """
    self._objective_func(self.A, self.X, self.y)

    print("optimizing...")
    ret = optimize.minimize(
      self._objective_func,
      self.A,
      args=(self.X, self.y),
      method="CG",
      jac=self._grad,
      options={'maxiter': max_iters, 'disp': True},
    )
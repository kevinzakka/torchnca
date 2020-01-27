"""A pytorch implementation of NCA.

Ref: https://www.cs.toronto.edu/~hinton/absps/nca.pdf
"""

import numpy as np
import torch


class NCA:
  """Neighbourhood Components Analysis.
  """
  def __init__(self, dim=None, init="random", max_iters=500, tol=1e-5):
    """Constructor.

    Args:
      dim (int): The dimension of the the learned
        linear map A. If no dimension is provided,
        we assume a square matrix A of same dimension
        as the input data. For small values of `dim`
        (i.e. 2, 3), NCA will perform dimensionality
        reduction.
      init (str): The type of initialization to use for
        the matrix A.
          - `random`: A = N(O, I)
          - `identity`: A = I
      max_iters (int): The maximum number of iterations
        to run the optimization for.
      tol (float): The tolerance for convergence. If the
        difference between consecutive solutions is within
        this number, optimization is terminated.
    """
    self.dim = dim
    self.init = init
    self.max_iters = max_iters
    self.tol = tol

  def __call__(self, X):
    """Apply the learned linear map to the input.
    """
    return torch.mm(X, torch.t(self.A))

  def _init_transformation(self):
    """Initialize the linear transformation A.
    """
    if self.dim is None:
      self.dim = self.num_dims
    if self.init == "random":
      self.A = torch.randn(self.dim, self.num_dims, requires_grad=True, device=self.device)
    elif self.init == "identity":
      self.A = torch.eye(self.dim, self.num_dims, requires_grad=True, device=self.device)
    else:
      raise ValueError("[!] {} initialization is not supported.".format(init))

  def _pairwise_l2_sq(self, X):
    """Compute pairwise squared Euclidean distances.
    """
    dot = torch.mm(X, torch.t(X))
    norm_sq = torch.diag(dot)
    dist = norm_sq[None, :] - 2*dot + norm_sq[:, None]
    dist = torch.clamp(dist, min=0)  # replace negative values with 0
    dist[dist != dist] = 0  # replace nan values with 0
    return dist

  def _softmax(self, X):
    """Compute row-wise softmax.
    """
    exp = torch.exp(X)
    # zero out diagonal
    mask = torch.eye(X.shape[0], device=self.device).byte()
    mask = (~mask).float()
    exp = exp * mask
    return exp / exp.sum(dim=1)

  def loss(self, X, y_mask):
    # compute pairwise squared Euclidean distances
    # in transformed space
    embedding = torch.mm(X, torch.t(self.A))
    distances = self._pairwise_l2_sq(embedding)

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
    p_ij_mask = p_ij * y_mask.float()

    # sum over js to compute p_i
    p_i = p_ij_mask.sum(dim=1)

    # compute expected number of points
    # correctly classified by summing
    # over all p_i's.
    # to maximize the above expectation
    # we can negate it and feed it to 
    # a minimizer
    loss = -torch.log(p_i).sum()

    return loss

  def train(self, X, y, batch_size=64):
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
    self.device = torch.device("cuda" if X.is_cuda else "cpu")

    # initialize the linear transformation matrix A
    self._init_transformation()

    optimizer = torch.optim.SGD([self.A], lr=1e-4, momentum=0.9)
    iters_per_epoch = int(np.ceil(self.num_train / batch_size))
    i_global = 0
    for epoch in range(self.max_iters):
      A_prev = optimizer.param_groups[0]['params'][0].clone()
      for i in range(iters_per_epoch):
        # grab batch
        X_batch = X[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]

        # compute pairwise boolean class matrix
        y_mask = y_batch[:, None] == y_batch[None, :]

        # compute loss and take gradient step
        optimizer.zero_grad()
        loss = self.loss(X_batch, y_mask)
        loss.backward()
        optimizer.step()

        i_global += 1
        if not i_global % 25:
          print("epoch: {} - loss: {:.2f}".format(epoch+1, loss.item()))

      # check if within convergence
      A_curr = optimizer.param_groups[0]['params'][0]
      if torch.all(torch.abs(A_prev - A_curr) < self.tol):
        print("[*] Optimization converged.")
        break

    self.A = optimizer.param_groups[0]['params'][0].clone()
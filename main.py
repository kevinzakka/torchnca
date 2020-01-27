"""NCA for linear dimensionality reduction.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from nca import NCA


def make_circle(r, num_samples):
  t = np.linspace(0, 2*np.pi, num_samples)
  xc, yc = 0, 0
  x = r*np.cos(t) + 0.2*np.random.randn(num_samples) + xc
  y = r*np.sin(t) + 0.2*np.random.randn(num_samples) + yc
  return x, y


def gen_data(num_samples, num_classes, mean, std, device):
  """Generates the data.
  """
  num_samples_per = num_samples // num_classes
  X = []
  y = []
  for i, r in enumerate(range(num_classes)):
    x1, x2 = make_circle(r+1+0.5, num_samples)
    x3 = std*np.random.randn(num_samples) + mean  # third dimension is noise
    X.append(np.stack([x1, x2, x3]))
    y.append(np.repeat(i, num_samples))
  X = np.concatenate(X, axis=1)
  y = np.concatenate(y)
  indices = list(range(X.shape[1]))
  np.random.shuffle(indices)
  X = X[:, indices]
  y = y[indices]
  X = X.T  # make it (N, D)
  X = torch.from_numpy(X).float().to(device)
  y = torch.from_numpy(y).long().to(device)
  return X, y


def plot(X, y):
  data = X.detach().cpu().numpy()
  labels = y.detach().cpu().numpy()
  plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Spectral)
  plt.grid(True)
  plt.show()


def main():
  np.random.seed(0)
  torch.cuda.manual_seed(0)
  device = torch.device("cuda")
  num_samples = 100

  X, y = gen_data(num_samples, 5, 0, 5, device)
  plot(X, y)

  nca = NCA(dim=2, init="random")
  nca.train(X, y, batch_size=64)

  # transform and plot
  X_tr = nca(X)
  plot(X_tr, y)


if __name__ == "__main__":
  main()
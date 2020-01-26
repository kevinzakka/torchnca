"""NCA for linear dimensionality reduction.
"""

import matplotlib.pyplot as plt
import numpy as np

from nca import NCA


def make_circle(r, num_samples):
  t = np.linspace(0, 2*np.pi, num_samples)
  xc, yc = 0, 0
  x = r*np.cos(t) + 0.2*np.random.randn(num_samples) + xc  
  y = r*np.sin(t) + 0.2*np.random.randn(num_samples) + yc
  return x, y


def gen_data(num_samples, num_classes=5, mean=0, std=5):
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
  return X, y
  


def main():
  np.random.seed(0)
  num_samples = 200

  X, y = gen_data(num_samples)

  # plot the data
  plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
  plt.grid(True)
  plt.show()

  nca = NCA(X, y, dim=2, init="identity")

  nca.train()


if __name__ == "__main__":
  main()
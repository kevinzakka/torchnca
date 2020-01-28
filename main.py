"""NCA for linear dimensionality reduction.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from nca import NCA
from sklearn.decomposition import PCA


def make_circle(r, num_samples):
  t = np.linspace(0, 2*np.pi, num_samples)
  xc, yc = 0, 0  # circle center coordinates
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
    # first two dimensions are that of a circle
    x1, x2 = make_circle(r+1.5, num_samples)
    # third dimension is Gaussian noise
    x3 = std*np.random.randn(num_samples) + mean
    X.append(np.stack([x1, x2, x3]))
    y.append(np.repeat(i, num_samples))
  X = np.concatenate(X, axis=1)
  y = np.concatenate(y)
  indices = list(range(X.shape[1]))
  np.random.shuffle(indices)
  X = X[:, indices]
  y = y[indices]
  X = X.T  # make it (N, D)
  return X, y


def plot(Xs, y, labels, save=None):
  fig, axes = plt.subplots(1, len(labels), figsize=(8, 4))
  for ax, X, lab in zip(axes, Xs, labels):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    ax.title.set_text(lab)
  if save is not None:
    plt.savefig("./assets/{}".format(save), format="png", dpi=300)
    plt.tight_layout()
  plt.show()


def main(args):
  np.random.seed(args.seed)
  if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
  else:
    print("[*] Using cpu.")
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

  num_samples = 200
  X, y = gen_data(num_samples, 5, 0, args.sigma, device)

  # plot first two dimensions of original data
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
  plt.show()

  # fit PCA
  pca = PCA(n_components=2)
  pca.fit(X)
  X_pca = pca.transform(X)

  # fit NCA
  X = torch.from_numpy(X).float().to(device)
  y = torch.from_numpy(y).long().to(device)
  nca = NCA(dim=2, init="identity", max_iters=500, tol=1e-5)
  nca.train(X, y, batch_size=256, lr=1e-6, momentum=0.9)
  X_nca = nca(X).detach().cpu().numpy()
  
  # plot PCA vs NCA
  y = y.detach().cpu().numpy()
  X = X.detach().cpu().numpy()
  plot([X_nca, X_pca], y, ["nca", "pca"])
  
  A = nca.A.detach().cpu().numpy()
  print("\nSolution: \n", A)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0, help="The rng seed.")
  parser.add_argument("--sigma", type=float, default=5, help="The standard deviation of the Gaussian noise.")
  parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1'], default=False, help="Whether to show GUI.")
  args, unparsed = parser.parse_known_args()
  main(args)
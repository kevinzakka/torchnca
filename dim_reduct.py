"""NCA for linear dimensionality reduction.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from nca import NCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    x1, x2 = make_circle(r+1.5, num_samples_per)
    # third dimension is Gaussian noise
    x3 = std*np.random.randn(num_samples_per) + mean
    X.append(np.stack([x1, x2, x3]))
    y.append(np.repeat(i, num_samples_per))
  X = np.concatenate(X, axis=1)
  y = np.concatenate(y)
  indices = list(range(X.shape[1]))
  np.random.shuffle(indices)
  X = X[:, indices]
  y = y[indices]
  X = X.T  # make it (N, D)
  return X, y


def plot(Xs, y, labels, save=None):
  fig, axes = plt.subplots(1, len(labels), figsize=(14, 4))
  for ax, X, lab in zip(axes, Xs, labels):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    ax.title.set_text(lab)
  if save is not None:
    filename = "./assets/{}".format(save)
    plt.savefig(filename, format="png", dpi=300, bbox_inches='tight')
  plt.show()


def main(args):
  np.random.seed(args.seed)
  if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
  else:
    print("[*] Using cpu.")
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

  num_samples = 300
  X, y = gen_data(num_samples, 5, 0, args.sigma, device)
  print("data", X.shape)

  # plot first two dimensions of original data
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
  plt.show()

  # fit PCA
  pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
  X_pca = pipeline.fit_transform(X)

  # fit LDA
  X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)

  # fit NCA
  X = torch.from_numpy(X).float().to(device)
  y = torch.from_numpy(y).long().to(device)
  nca = NCA(dim=2, init=args.init, max_iters=1000, tol=1e-5)
  nca.train(X, y, batch_size=None, weight_decay=10)
  X_nca = nca(X).detach().cpu().numpy()
  
  # plot PCA vs NCA
  y = y.detach().cpu().numpy()
  plot([X_nca, X_pca, X_lda], y, ["nca", "pca", "lda"], save="res.png")
  
  A = nca.A.detach().cpu().numpy()
  print("\nSolution: \n", A)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0, help="The rng seed.")
  parser.add_argument("--sigma", type=float, default=5, help="The standard deviation of the Gaussian noise.")
  parser.add_argument("--init", type=str, default="identity", help="Which initialization to use.")
  parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1'], default=False, help="Whether to show GUI.")
  args, unparsed = parser.parse_known_args()
  main(args)

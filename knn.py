"""NCA + kNN vs. vanilla kNN.

TODO:
  1. Tune the value of `dim` in NCA.
  2. Tune the number of neighbours `k` in kNN.
"""

import argparse
import numpy as np
import time
import torch

from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from nca import NCA


def main(args):
  np.random.seed(args.seed)
  use_cuda = args.cuda and torch.cuda.is_available()
  if use_cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
  else:
    print("[*] Using cpu.")
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

  # load the MNIST dataset
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])
  mnist_data = datasets.MNIST('./data', train=True, transform=transform)

  # split into train and test
  X_train = mnist_data.data[:50000]
  y_train = mnist_data.targets[:50000]
  X_test = mnist_data.data[50000:]
  y_test = mnist_data.targets[50000:]

  # flatten to (N, D)
  X_train = X_train.view(X_train.shape[0], -1).float() / 255.
  X_test = X_test.view(X_test.shape[0], -1).float() / 255.
  X_test = X_test.cpu().numpy()
  y_test = y_test.cpu().numpy()

  # NCA + kNN
  nca = NCA(dim=32, init=args.init, max_iters=70, tol=1e-5)
  nca.train(X_train, y_train, batch_size=512, weight_decay=10, lr=1e-4, normalize=False)
  A = nca.A.detach().cpu().numpy()
  X_train = X_train.cpu().numpy()
  y_train = y_train.cpu().numpy()
  X_train_embed = X_train @ A.T
  X_test_embed = X_test @ A.T
  knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
  knn.fit(X_train_embed, y_train)
  tic = time.time()
  predictions = knn.predict(X_test_embed)
  toc = time.time()
  nca_time = toc - tic
  nca_error = 1 - accuracy_score(predictions, y_test)
  nca_bytes = X_train_embed.size * X_train_embed.itemsize
  print("nca knn - time: {:.2f} - error: {:.2f} - storage: {:.2f} Mb".format(
    nca_time, 100 * nca_error, nca_bytes*1e-6))

  # raw kNN
  knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
  knn.fit(X_train, y_train)
  tic = time.time()
  predictions = knn.predict(X_test)
  toc = time.time()
  vanilla_time = toc - tic
  vanilla_error = 1 - accuracy_score(predictions, y_test)
  vanilla_bytes = X_train.size * X_train.itemsize
  print("vanilla knn - time: {:.2f} - error: {:.2f} - storage: {:.2f} Mb".format(
    vanilla_time, 100*vanilla_error, vanilla_bytes*1e-6))

  speedup = vanilla_time / nca_time
  print("speedup: {:.2f}x".format(speedup))



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0, help="The rng seed.")
  parser.add_argument("--init", type=str, default="random", help="Which initialization to use.")
  parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1'], default=False, help="Whether to show GUI.")
  args, unparsed = parser.parse_known_args()
  main(args)

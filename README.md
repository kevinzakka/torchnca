# Neighbourhood Components Analysis

A PyTorch implementation of [Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf) by *J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov*.

NCA learns a linear transformation of the dataset such that the expected leave-one-out performance of kNN in the transformed space is maximized.

## API

```python
# instantiate nca object and initialize with
# an identity matrix
nca = NCA(dim=2, init="identity")

# fit an nca model to a dataset
nca.train(X, y, batch_size=64)

# apply the learned linear map to the data
X_nca = nca(X)
```

## Dimensionality Reduction

We generate a 3-D dataset where the first 2 dimensions are concentric rings and the third dimension is Gaussian noise. We plot the result of PCA and NCA with 2 components.

<p align="center">
 <img src="./assets/res.png" width="80%">
</p>

Notice how PCA has failed to project out the noise, a result of a high noise variance in the third dimension.

## Notes

* The learning rate, batch size and initialization strategy are critical for convergence. You need to tune them to obtain a sensible solution.
* Lowering the noise variance of the third dimension (e.g. `0.1`) makes it easier for PCA to tease out the projection.

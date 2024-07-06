# Non-negative Matrix Factorization

[![Build Status](https://github.com/john-waczak/MLJNonnegativeMatrixFactorization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/john-waczak/MLJNonnegativeMatrixFactorization.jl/actions/workflows/CI.yml?query=branch%3Amain)


A Julia package for Non-negative Matrix Factorization based on [the method introduced by Lee and Seung](https://www.nature.com/articles/44565) and compliant with the MLJ model interface.


# Using the Package

To train an NMF model, first load MLJ and this package

```julia
using MLJ, MLJNonnegativeMatrixFactorization
```

The NMF can then be instantiated in the usual way for an unsupervised model:

```julia
nmf = NMF()
mach = machine(nmf, df)
fit!(mach)
```

Calling `fitted_params` on the trained machine will return a tuple containing the factor matrix (a.k.a. endmembers) `W` and the factor loading matrix (a.k.a. abundances) `H`.

The NMF can be used for feature extraction via `W` or for dimensionality reduction to the factor loadings via

```julia
Ĥ = MLJ.transform(mach, X)
```

module MLJNonnegativeMatrixFactorization

using Random
using LinearAlgebra
using Statistics, MultivariateStats
using Distances
using MLJModelInterface
import MLJBase



include("cost-functions.jl")

mk_rng(rng::AbstractRNG) = rng
mk_rng(int::Integer) = Random.MersenneTwister(int)

include("nmf-base.jl")
include("nmf-mlj.jl")

export NMF


# ---------------------------------------------------------------------------
# ---------- DOCS -----------------------------------------------------------
# ---------------------------------------------------------------------------

MLJModelInterface.metadata_pkg.(
    [NMF,],
    name = "MLJNonnegativeMatrixFactorization",
    uuid = "5fbdd59d-4608-42a7-928b-f02c7ff8d663", # see your Project.toml
    url  = "https://github.com/john-waczak/MLJNonnegativeMatrixFactorization.jl",
    julia = true,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)


MLJModelInterface.metadata_model(
    NMF,
    input_scitype = MLJModelInterface.Table(MLJModelInterface.Continuous),
    output_scitype  = MLJModelInterface.Table(MLJModelInterface.Continuous),
    supports_weights = false,
	  load_path    = "MLJNonnegativeMatrixFactorization.NMF"
)




const DOC_GTM = ""*
    ""*
    ""



"""
$(MLJModelInterface.doc_header(NMF))

MLJNonnegativeMatrixFactorization implements [Nonnegative Matrix Factorization](https://www.nature.com/articles/44565),
  Nature; Lee, D.D. & Seung, H.S.; (1999):
  \"Learning the parts of objects by non-negative matrix factorization\"


# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)
where
- `X`: `Table` of input features whose columns are of scitype `Continuous.`

Train the machine with `fit!(mach, rows=...)`.

# Hyper-parameters
- `k=3`: Number of factors (endmembers) for decomposition.
- `cost=:Euclidean`:  Cost function used for fitting. One of `:Euclidean, :KL, :L21`
- `λw1=0.0`: L1 regularization strength for `W` matrix.
- `λw2=0.0`: L2 regularization strength for `W` matrix.
- `λh1=0.0`: L1 regularization strength for `H` matrix.
- `λh2=0.0`: L2 regularization strength for `H` matrix.
- `normalize_abundance=true`: If `true`, normlize abundance matrix H to unity.
- `tol=0.1`: Tolerance used for determining convergence during fitting.
- `nconverged=4`: Number of steps to repeat at/below `tol` before NMF is considered converged.
- `maxiters=100`: Maximum number of algorithm iterations.
- `rng=123`: Random number seed or random number generator used for weight initialization.

# Operations
- `transform(mach, X)`: returns the projected coordinates corresponding to the estimated abundances `h_k`.

# Fitted parameters
The fields of `fitted_params(mach)` are:

- `W`: The factor (endmember) vectors
- `H`: The factor loadings (abundances)

# Report
The fields of `report(mach)` are:
- `cost`: Final value of cost function
- `converged`: is `true` if the convergence critera were met before reaching `niter`

# Examples
```
using MLJ
nmf = @load GTM pkg=GenerativeTopographicMapping
model = nmf()
⋮
mach = machine(model, X) |> fit!
H̃ = transform(mach, X)

```
"""
NMF



end

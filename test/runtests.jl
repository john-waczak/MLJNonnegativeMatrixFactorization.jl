using MLJNonnegativeMatrixFactorization
using Test

using MLJBase
using StableRNGs  # should add this for later
using MLJTestIntegration
using Tables
using Distances
using LinearAlgebra
using Distributions
using Random

rng = StableRNG(1234)

const m = 10
const n = 100
const k = 3



@testset "cost-functions.jl" begin
    # test with square matrix to make things easy
    X = ones(m, n)

    W = ones(m, k) ./ 3
    H = ones(k, n)

    # perfect reconstruction should yield zero cost
    @test MLJNonnegativeMatrixFactorization.frob(W, H, X) == 0.0
    @test MLJNonnegativeMatrixFactorization.kl_div(W, H, X) == 0.0
    @test MLJNonnegativeMatrixFactorization.L21(W, H, X) == 0.0

    # λ = 0.1
    # @test MLJNonnegativeMatrixFactorization.L1_2(W, H, X,λ) - λ * sum(sqrt.(H)) ==  0.0
end



# Generate test dataset to be used throughout
f = Dirichlet([0.5, 0.5, 0.5])
E = rand(rng, m,k)
A = rand(rng, f, n)
X = E*A

@testset "nmf-base.jl" begin
    nmf = MLJNonnegativeMatrixFactorization.NMFBase(m,n,k,rng)

    @test size(nmf.W) == (m,k)
    @test size(nmf.H) == (k,n)


    converged, cost = MLJNonnegativeMatrixFactorization.fit_frob!(nmf, X, verbose=false, maxiters=5000, tol=1e-5,)

    @test all(nmf.W .≥ 0.0)
    @test all(nmf.H .≥ 0.0)
    @test all(sum(nmf.H, dims=1) .≈ 1.0)
    # @test  isapprox(nmf.W*nmf.H, X, atol=0.01)

    nmf = MLJNonnegativeMatrixFactorization.NMFBase(m,n,k,rng)
    converged, cost = MLJNonnegativeMatrixFactorization.fit_frob!(nmf, X,
                                                                  λw1=0.01, λw2=0.1,
                                                                  λh1=0.1, λh2=0.01,
                                                                  verbose=false,
                                                                  maxiters=1000,
                                                                  tol=1e-5)


    nmf = MLJNonnegativeMatrixFactorization.NMFBase(m,n,k,rng)
    W2 = deepcopy(nmf.W)
    H2 = deepcopy(nmf.H)

    converged, cost = MLJNonnegativeMatrixFactorization.fit_frob!(nmf, X,
                                                                  verbose=false,
                                                                  maxiters=500,
                                                                  tol=1e-19,
                                                                  normalize_abundance=false)
    # fit mse NMF using NMF.jl and ompare
    let
        import NMF as NMF2
        NMF2.solve!(NMF2.MultUpdate{Float64}(obj=:mse, maxiter=500, verbose=false), X, W2, H2)

        cost2 = MLJNonnegativeMatrixFactorization.frob(W2, H2, X)
        @test isapprox(cost, cost2, rtol=1e-3)
    end


    nmf = MLJNonnegativeMatrixFactorization.NMFBase(m,n,k,rng)
    converged, cost = MLJNonnegativeMatrixFactorization.fit_kl!(nmf, X, verbose=false, maxiters=1000, tol=1e-5,)

    @test all(nmf.W .≥ 0.0)
    @test all(nmf.H .≥ 0.0)
    @test all(sum(nmf.H, dims=1) .≈ 1.0)


    nmf = MLJNonnegativeMatrixFactorization.NMFBase(m,n,k,rng)
    converged, cost = MLJNonnegativeMatrixFactorization.fit_kl!(nmf, X,
                                                                λw1=0.01, λw2=0.1,
                                                                λh1=0.1, λh2=0.01,
                                                                verbose=false,
                                                                maxiters=1000,
                                                                tol=1e-5)


    nmf = MLJNonnegativeMatrixFactorization.NMFBase(m,n,k,rng)
    W2 = deepcopy(nmf.W)
    H2 = deepcopy(nmf.H)

    converged, cost = MLJNonnegativeMatrixFactorization.fit_kl!(nmf, X,
                                                                verbose=false,
                                                                maxiters=500,
                                                                tol=1e-19,
                                                                normalize_abundance=false)
    let
        import NMF as NMF2
        NMF2.solve!(NMF2.MultUpdate{Float64}(obj=:div, maxiter=500, verbose=false), X, W2, H2)

        cost2 = MLJNonnegativeMatrixFactorization.kl_div(W2, H2, X)
        @test isapprox(cost, cost2, rtol=1e-3)
    end

    nmf = MLJNonnegativeMatrixFactorization.NMFBase(m,n,k,rng)
    converged, cost = MLJNonnegativeMatrixFactorization.fit_L21!(nmf, X, verbose=false, maxiters=1000, tol=1e-5,)

    @test all(nmf.W .≥ 0.0)
    @test all(nmf.H .≥ 0.0)
    @test all(sum(nmf.H, dims=1) .≈ 1.0)

    nmf = MLJNonnegativeMatrixFactorization.NMFBase(m,n,k,rng)
    converged, cost = MLJNonnegativeMatrixFactorization.fit_L21!(nmf, X,
                                                                λw1=0.01, λw2=0.1,
                                                                λh1=0.1, λh2=0.01,
                                                                verbose=false,
                                                                 maxiters=1000,
                                                                 tol=1e-5)
end



@testset "nmf-mlj.jl" begin
    # EUCLIDEAN
    model = NMF(k=k, cost=:Euclidean, rng=rng)
    Xtable = Tables.table(X')
    m = machine(model, Xtable)
    fit!(m, verbosity=0)

    fp = fitted_params(m)
    Ŵ  = fp[:W]
    Ĥ  = fp[:H]

    @test all(Ĥ .≥ 0.0)
    @test all(Ŵ .≥ 0.0)

    X̃ = MLJBase.transform(m, X)
    @test length(keys(X̃)) == k


    # KL-DIVERGENCE
    model = NMF(k=k, cost=:KL, rng=rng)
    Xtable = Tables.table(X')
    m = machine(model, Xtable)
    fit!(m, verbosity=0)

    fp = fitted_params(m)
    Ŵ  = fp[:W]
    Ĥ  = fp[:H]

    @test all(Ĥ .≥ 0.0)
    @test all(Ŵ .≥ 0.0)


    # L21
    model = NMF(k=k, cost=:L21, rng=rng)
    Xtable = Tables.table(X')
    m = machine(model, Xtable)
    fit!(m, verbosity=0)

    fp = fitted_params(m)
    Ŵ  = fp[:W]
    Ĥ  = fp[:H]

    @test all(Ĥ .≥ 0.0)
    @test all(Ŵ .≥ 0.0)


    # EUCLIDEAN - w/ regularization
    model = NMF(k=k, cost=:Euclidean, rng=rng, λw1=0.01, λw2=0.1, λh1=0.1, λh2=0.01)
    Xtable = Tables.table(X')
    m = machine(model, Xtable)
    fit!(m, verbosity=0)

    fp = fitted_params(m)
    Ŵ  = fp[:W]
    Ĥ  = fp[:H]

    @test all(Ĥ .≥ 0.0)
    @test all(Ŵ .≥ 0.0)

    # KL-DIVERGENCE
    model = NMF(k=k, cost=:KL, rng=rng, λw1=0.01, λw2=0.1, λh1=0.1, λh2=0.01)
    Xtable = Tables.table(X')
    m = machine(model, Xtable)
    fit!(m, verbosity=0)

    fp = fitted_params(m)
    Ŵ  = fp[:W]
    Ĥ  = fp[:H]

    @test all(Ĥ .≥ 0.0)
    @test all(Ŵ .≥ 0.0)


    # L21
    model = NMF(k=k, cost=:L21, rng=rng, λw1=0.01, λw2=0.1, λh1=0.1, λh2=0.01)
    Xtable = Tables.table(X')
    m = machine(model, Xtable)
    fit!(m, verbosity=0)

    fp = fitted_params(m)
    Ŵ  = fp[:W]
    Ĥ  = fp[:H]

    @test all(Ĥ .≥ 0.0)
    @test all(Ŵ .≥ 0.0)
end



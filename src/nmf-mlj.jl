mutable struct NMF <: MLJMdelInterface.Unsupervised
    k::Int
    cost::Symbol
    λw1::Float64
    λw2::Float64
    λh1::Float64
    λh2::Float64
    normalize_abundance::Bool
    tol::Float64
    nconverged::Int
    maxiters::Int
    rng::Any
end


function NMF(; k=3, cost=:Euclidean, λw1=0.0, λw2=0.0, λh1=0.0, λh2=0.0, normalize_abundance=true, tol=1e-3, nconverged=4, maxiters=1000, rng=123)
    model = NMF(k, cost, λw1, λw2, λh1, λh2, normalize_abundance, tol, nconverged, maxiters, mk_rng(rng))
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end



function MLJModelInterface.clean!(m::NMF)
    warning =""

    if m.k ≤ 0
        warning *= "Parameter `k` expected to be positive, resetting to 3\n"
        m.k = 3
    end

    if !(m.cost ∈ [:Euclidean, :KL, :L21])
        warning *= "Parameter `cost` must be one of `:Euclidean, :KL, :L21`, resetting to `:Euclidean`\n"
        m.cost = :Euclidean
    end

    if m.λw1 < 0
        warning *= "Parameter `λw1` expected to be non-negative, resetting to 0.0\n"
        m.λw1 = 0.0
    end

    if m.λw2 < 0
        warning *= "Parameter `λw2` expected to be non-negative, resetting to 0.0\n"
        m.λw2 = 0.0
    end

    if m.λh1 < 0
        warning *= "Parameter `λh1` expected to be non-negative, resetting to 0.0\n"
        m.λh1 = 0.0
    end

    if m.λh2 < 0
        warning *= "Parameter `λh2` expected to be non-negative, resetting to 0.0\n"
        m.λh2 = 0.0
    end

    if m.tol ≤ 0
        warning *= "Parameter `tol` expected to be positive, resetting to 1e-3\n"
        m.tol = 1e-3
    end


    if m.maxiters ≤ 0
        warning *= "Parameter `maxiters` expected to be positive, resetting to 1000\n"
        m.maxiters = 1000
    end

    if m.nconverged < 0
        warning *= "Parameter `nconverged` expected to be non-negative, resetting to 4\n"
        m.nconverged = 4
    end

    return warning
end




function MLJModelInterface.fit(m::NMF, verbosity, Datatable)
    # assume that X is a table
    X = MLJModelInterface.matrix(Datatable)'
    (M,N) = size(X)

    if verbosity > 0
        verbose = true
    else
        verbose = false
    end

    # 1. build the GTM
    nmf = NMFBase(M, N, m.k, m.rng)


    fit! = fit_frob!
    # check if we should use a different function
    if m.cost == :KL
        fit! = fit_kl!
    elseif m.cost == :L21
        fit! = fit_L21!
    else
        fit! = fit_frob!
    end


    # 2. Fit the NMF
    converged, cost = fit!(
        nmf,
        X;
        λw1=m.λw1, λw2=m.λw2,
        λh1=m.λh1, λh2=m.λh2,
        normalize_abundance = m.normalize_abundance,
        tol=m.tol,
        nconverged=m.nconverged,
        maxiters=m.maxiters,
        verbose=verbose,
    )

    # 3. Collect results
    cache = nothing

    report = (;
              :converged => converged,
              :cost => cost,
             )

    return (nmf, cache, report)
end

MLJModelInterface.fitted_params(m::NMF, fitresult) = (W=fitresult.W, H=fitresult.H)


function MLJModelInterface.transform(m::NMF, fitresult, Data_new)
    fitresult.H
    names = [Symbol("h_$(i)") for i ∈ 1:m.k]

    return MLJModelInterface.table(fitresult.H'; names=names)
end




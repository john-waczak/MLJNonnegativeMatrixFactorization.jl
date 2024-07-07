mutable struct NMFBase{T1 <: AbstractArray, T2 <: AbstractArray,}
    m::Int  # number of features
    n::Int  # number of samples
    k::Int  # number of factors
    W::T1   # endmember matrix
    H::T2   # abundance matrix
end


# this can be updated to include more initialization options
# e.g. SVD, VCA, etc...
function NMFBase(m,n,k, rng=mk_rng(123))
    # for now, just randomly initialize the matrices
    W = rand(rng, m,k)
    H = rand(rng, k,n)

    return NMFBase(m, n, k, W, H)
end



#-------------------
#-- FIT FUNCTIONS --
#-------------------
function fit_frob!(nmf::NMFBase, X;
                   λw1=0.0, λw2=0.0,
                   λh1=0.0, λh2=0.0,
                   normalize_abundance=true,
                   tol=1e-3,
                   nconverged=5,
                   maxiters=100,
                   verbose=false,
                   )
    # verify shape of dataset
    @assert size(X) == (nmf.m,nmf.n)

    cost = 0.0
    cost_prev = 0.0
    nclose = 0
    converged = false

    # preallocate matrices
    WH = nmf.W * nmf.H

    WtX = nmf.W' * X
    WtWH = nmf.W'*WH

    XHt = X*nmf.H'
    WHHt = WH*nmf.H'

    if normalize_abundance
        for n ∈ 1:nmf.n
            nmf.H[:,n] .= nmf.H[:,n] ./ sum(nmf.H[:,n])
        end
    end

    # update loop
    for i ∈ 1:maxiters
        # 1a. Update product matrices
        mul!(WtX, nmf.W', X)
        mul!(WtWH, nmf.W', WH)

        # 1b. Update H
        Threads.@threads for idx ∈ 1:length(nmf.H)
            @inbounds nmf.H[idx] *= (WtX[idx])/(WtWH[idx] + λh1 + λh2*nmf.H[idx])
        end

        # 1c. (optional) Normalize columns of H
        if normalize_abundance
            for n ∈ 1:nmf.n
                nmf.H[:,n] .= nmf.H[:,n] ./ sum(nmf.H[:,n])
            end
        end
        mul!(WH, nmf.W, nmf.H)

        # 2a. Update product matrics
        mul!(XHt, X, nmf.H')
        mul!(WHHt, WH, nmf.H')

        # 2b. Update
        Threads.@threads for idx ∈ 1:length(nmf.W)
            @inbounds nmf.W[idx] *= (XHt[idx])/(WHHt[idx] + λw1 + λw2*nmf.W[idx])
        end

        mul!(WH, nmf.W, nmf.H)

        # 3. Check convergence
        if i == 1
            cost = frob(nmf.W, nmf.H, X)
        else
            cost_prev = cost
            cost = frob(nmf.W, nmf.H, X)

            diff= abs(cost - cost_prev)/min(abs(cost), abs(cost_prev))

            if diff <= tol
                # increment the number of "close" differences
                nclose += 1
            end

            if nclose == nconverged
                converged = true
                break
            end
        end

        if i%10==0
            if verbose
                println("iter: $(i), Cost = $(cost)")
            end
        end
    end

    return converged, cost
end



function fit_kl!(nmf::NMFBase, X;
                   λw1=0.0, λw2=0.0,
                   λh1=0.0, λh2=0.0,
                   normalize_abundance=true,
                   tol=1e-3,
                   nconverged=5,
                   maxiters=100,
                   verbose=false,
                   )
    # verify shape of dataset
    @assert size(X) == (nmf.m,nmf.n)

    cost = 0.0
    cost_prev = 0.0
    nclose = 0
    converged = false

    # preallocate matrices
    WH = nmf.W * nmf.H
    Q = X ./ WH

    ΣH = sum(nmf.H, dims=2)
    ΣW = sum(nmf.W, dims=1)

    WtQ = nmf.W'*Q
    QHt = Q*nmf.H'

    if normalize_abundance
        for n ∈ 1:nmf.n
            nmf.H[:,n] .= nmf.H[:,n] ./ sum(nmf.H[:,n])
        end
    end


    # update loop
    for i ∈ 1:maxiters
        # 1a. Update product matrices
        @inbounds for idx ∈ 1:length(Q)
            Q[idx] = X[idx] / (WH[idx] + eps(eltype(WH)))
        end

        mul!(WtQ, nmf.W', Q)
        sum!(fill!(ΣW, 0), nmf.W)

        # 1b. Update H
        @inbounds for n ∈ 1:nmf.n, k ∈ 1:nmf.k
            nmf.H[k,n] *= max(WtQ[k,n] / (ΣW[k] + λh1 + λh2*nmf.H[k,n]), eps(eltype(WH)))
        end

        # 1c. (optional) Normalize columns of H
        if normalize_abundance
            for n ∈ 1:nmf.n
                nmf.H[:,n] .= nmf.H[:,n] ./ sum(nmf.H[:,n])
            end
        end
        mul!(WH, nmf.W, nmf.H)

        # 2a. Update product matrics
        @inbounds for idx ∈ 1:length(Q)
            Q[idx] = X[idx] / (WH[idx] + eps(eltype(WH)))
        end

        mul!(QHt, Q, nmf.H')
        sum!(fill!(ΣH, 0), nmf.H)

        # 2b. Update W
        @inbounds for k ∈ 1:nmf.k, m ∈ 1:nmf.m
            nmf.W[m,k] *= max(QHt[m,k] / (ΣH[k] + λw1 + λw2*nmf.W[m,k]), eps(eltype(WH)))
        end

        mul!(WH, nmf.W, nmf.H)

        # 3. Check convergence
        if i == 1
            cost = kl_div(max.(nmf.W, eps(eltype(WH))), max.(nmf.H, eps(eltype(WH))), X)
        else
            cost_prev = cost
            cost = kl_div(max.(nmf.W, eps(eltype(WH))), max.(nmf.H, eps(eltype(WH))), X)
            # cost = kl_div(nmf.W, nmf.H, X)

            diff= abs(cost - cost_prev)/min(abs(cost), abs(cost_prev))
            if diff <= tol
                # increment the number of "close" differences
                nclose += 1
            end

            if nclose == nconverged
                converged = true
                break
            end
        end

        if i%10==0
            if verbose
                println("iter: $(i), Cost = $(cost)")
            end
        end
    end

    return converged, cost
end



function fit_L21!(nmf::NMFBase, X;
                   λw1=0.0, λw2=0.0,
                   λh1=0.0, λh2=0.0,
                   normalize_abundance=true,
                   tol=1e-3,
                   nconverged=5,
                   maxiters=100,
                   verbose=false,
                   )
    # verify shape of dataset
    @assert size(X) == (nmf.m,nmf.n)

    cost = 0.0
    cost_prev = 0.0
    nclose = 0
    converged = false

    D = Diagonal(1.0 ./ sqrt.(sum((X .- nmf.W*nmf.H).^2, dims=1)[:]))
    WtXD = nmf.W'*X*D
    WtWHD = nmf.W'*nmf.W*nmf.H*D
    XDHt = X*D*nmf.H'
    WHDHt = nmf.W*nmf.H*D*nmf.H'

    if normalize_abundance
        for n ∈ 1:nmf.n
            nmf.H[:,n] .= nmf.H[:,n] ./ sum(nmf.H[:,n])
        end
    end

    # update loop
    for i ∈ 1:maxiters
        # 1a. Update product matrices
        D .= Diagonal(1.0 ./ sqrt.(sum((X .- nmf.W*nmf.H).^2, dims=1)[:]))
        WtXD .= nmf.W'*X*D
        WtWHD .= nmf.W'*nmf.W*nmf.H*D

        # 1b. Update H
        Threads.@threads for idx ∈ 1:length(nmf.H)
            @inbounds nmf.H[idx] *= (WtXD[idx])/(WtWHD[idx] + λh1 + λh2*nmf.H[idx])
        end

        # 1c. (optional) Normalize columns of H
        if normalize_abundance
            for n ∈ 1:nmf.n
                nmf.H[:,n] .= nmf.H[:,n] ./ sum(nmf.H[:,n])
            end
        end

        # 2a. Update product matrics
        D .= Diagonal(1.0 ./ sqrt.(sum((X .- nmf.W*nmf.H).^2, dims=1)[:]))
        XDHt .= X*D*nmf.H'
        WHDHt .= nmf.W*nmf.H*D*nmf.H'

        # 2b. Update
        Threads.@threads for idx ∈ 1:length(nmf.W)
            @inbounds nmf.W[idx] *= (XDHt[idx])/(WHDHt[idx] + λw1 + λw2*nmf.W[idx])
        end

        # 3. Check convergence
        if i == 1
            cost = L21(nmf.W, nmf.H, X)
        else
            cost_prev = cost
            cost = L21(nmf.W, nmf.H, X)

            diff= abs(cost - cost_prev)/min(abs(cost), abs(cost_prev))

            if diff <= tol
                # increment the number of "close" differences
                nclose += 1
            end

            if nclose == nconverged
                converged = true
                break
            end
        end

        if i%10==0
            if verbose
                println("iter: $(i), Cost = $(cost)")
            end
        end
    end

    return converged, cost
end



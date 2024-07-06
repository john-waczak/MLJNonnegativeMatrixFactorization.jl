using Distances


# X - (m × n)
# W - (m × k)
# H - (k × n)


function frob(W, H, X)
    return sum((X .- W*H).^2)
end


function kl_div(W, H, X)
    return sum((X .* log.(X)) .- (X .* log.(W*H)) .- X .+ W*H)
end

function L21(W, H, X)
    return sum(sum((X .- W*H).^2, dims=1).^(0.5))
end


# function L1_2(W, H, X, λ)
#     return frob(W, H, X) .+ λ*sum(sqrt.(H))
# end

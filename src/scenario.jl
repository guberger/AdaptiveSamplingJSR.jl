# risk that the solution of a N-scenario convex program with Helly's dimension
# ζ does not satisfy the ϵ-chance-constraint version of the program
function risk_cc_convex(ϵ, ζ, N)
    q = ζ - 1
    D = Beta(N - q, q + 1)
    return cdf(D, 1 - ϵ)
end

# ϵ such that the solution of a N-scenario convex program with Helly's
# dimension ζ does not satisfies the ϵ-chance-constraint version of the program
# with risk 1 - β
function risk_cc_convex_inv(β, ζ, N)
    q = ζ - 1
    D = Beta(N - q, q + 1)
    return 1 - quantile(D, β)
end
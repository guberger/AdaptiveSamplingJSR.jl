module Experiment

using LinearAlgebra
using Random
using LaTeXStrings
using StatsPlots
using JuMP
using Gurobi
using MosekTools

const GUROBI_ENV = Gurobi.Env()
gurobi_opt() = Gurobi.Optimizer(GUROBI_ENV)
solverLP() = Model(optimizer_with_attributes(gurobi_opt, MOI.Silent()=>true))
mosek_opt() = Mosek.Optimizer()
solverSDP() = Model(optimizer_with_attributes(mosek_opt, MOI.Silent()=>true))

include("../src/AdaptiveSamplingJSR.jl")
const AS = AdaptiveSamplingJSR

Random.seed!(0)

nvar = 3
nmode = 3
γ_min = 0
γ_max = 10

A_list = [randn(nvar, nvar) for _ = 1:nmode]
display(A_list)

β = 0.05
ζ = nvar * (nvar + 1) / 2
C1 = 5
C2 = 2

#-------------------------------------------------------------------------------

samplesize1_list = 50:50:950
samplesize_total = 1000
expected_ratio = Tuple{Int,Float64}[]

for samplesize1 in samplesize1_list
    samplesize2 = samplesize_total - samplesize1
    ϵ = AS.risk_cc_convex_inv(0.5, 1, samplesize1 - ζ)
    area = ϵ * nmode * C1^(nvar / 4)
    s = AS.area_2sided_cap_inv(area, nvar)
    factor1 = 1 / s
    ϵ = AS.risk_cc_convex_inv(β, ζ, samplesize2)
    area = ϵ * nmode * C2^((nvar - 1) / 2)
    s = AS.area_2sided_cap_inv(area, nvar)
    factor2 = 1 / s
    push!(expected_ratio, (samplesize1, factor1 * factor2))
end

the_min = argmin(t -> t[2], expected_ratio)
display(the_min)
display(expected_ratio)

plt = plot(xlabel=L"N_1", ylabel="ratio")
plot!(plt, getindex.(expected_ratio, 1), getindex.(expected_ratio, 2))
scatter!(plt, [the_min[1]], [the_min[2]])
savefig(plt, "experiments/figures/example_3_3_heuristic_N1.png")
display(plt)

#-------------------------------------------------------------------------------

bounds = Tuple{Int,Float64}[]

for samplesize1 in samplesize1_list
    samplesize2 = samplesize_total - samplesize1
    
    trans_list = AS.Transition[]
    for _ = 1:samplesize1
        x = randn(nvar)
        q = rand(eachindex(A_list))
        y = A_list[q] * x
        push!(trans_list, AS.Transition(x, y))
    end
    
    @assert length(trans_list) == samplesize1
    γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, C1,
                                        γ_min, γ_max, solverLP,
                                        out=false)
    @assert isposdef(P)
    singvals = svdvals(P)
    κ = sqrt(prod(singvals) / minimum(singvals)^nvar)
    display(κ)
    Binv = cholesky(P).U
    B = inv(Binv)

    empty!(trans_list)
    for _ = 1:samplesize2
        x = randn(nvar)
        xp = B * x
        q = rand(eachindex(A_list))
        yp = A_list[q] * xp
        y = Binv * yp
        push!(trans_list, AS.Transition(x, y))
    end

    @assert length(trans_list) == samplesize2
    γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, C2,
                                        γ_min, γ_max, solverLP,
                                        out=false)
    @assert isposdef(P)
    singvals = svdvals(P)
    κ = sqrt(prod(singvals) / minimum(singvals)^nvar)
    display(κ)
    ϵ = AS.risk_cc_convex_inv(β, ζ, length(trans_list))
    area = ϵ * nmode * κ
    s = AS.area_2sided_cap_inv(area, nvar)
    factor = 1 / s
    push!(bounds, (samplesize1, γ * factor))
end

display(bounds)

plt = plot(xlabel=L"N_1", ylabel="bound")
plot!(plt, getindex.(bounds, 1), getindex.(bounds, 2))
savefig(plt, "experiments/figures/example_3_3_different_N1.png")
display(plt)

end # module
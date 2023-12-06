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

Random.seed!(1)

nvar = 4
nmode = 3
γ_min = 0
γ_max = 10

A_list = [randn(nvar, nvar) for _ = 1:nmode]

samplesize_min = 50
samplesize_max = 500
nsize = 2^5
samplesize_list_raw = range(samplesize_min, samplesize_max, length=nsize + 2)
samplesize_list = ceil.(Int, samplesize_list_raw[2:(end - 1)])
nexp = 2^5
ratios = Tuple{Int,Float64}[]
bounds = Tuple{Int,Vector{Float64},Float64,Vector{Float64}}[]
def_ratio = 5
def_bound1 = 5
def_bound2 = 5
def_prod = 5
β1s = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
β2 = 0.05
ζ1 = nvar * (nvar + 1) / 2
ζ2 = 1
C1 = 5
C2 = 1

for samplesize in samplesize_list
    for iexp in 1:nexp
        trans_list = AS.Transition[]
        for _ = 1:samplesize
            x = randn(nvar)
            q = rand(eachindex(A_list))
            y = A_list[q] * x
            push!(trans_list, AS.Transition(x, y))
        end
        γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, C1,
                                            γ_min, γ_max, solverLP,
                                            out=false)
        ratio = float(def_ratio)
        if !isposdef(P)
            push!(ratios, (samplesize, def_ratio))
            continue
        end
        γ_wb = AS.contraction_factor(A_list, P, solverSDP)
        ratio = γ_wb / γ
        # singvals = svdvals(P)
        # κ = prod(singvals) / minimum(singvals)
        display(cond(P))
        push!(ratios, (samplesize, ratio))
    end
    ϵ2 = AS.risk_cc_convex_inv(β2, ζ2, samplesize_max - samplesize)
    A2 = ϵ2 * nmode * C2^(1 - 1 / nvar)
    flag = true
    bound2 = def_bound2
    if A2 < 1
        bound2 = 1 / AS.area_2sided_cap_inv(A2, nvar)
        bound2 = min(bound2, def_bound2)
    else
        flag = false
    end
    bounds1 = fill(float(def_bound1), length(β1s))
    prods = fill(float(def_prod), length(β1s))
    for (i, β1) in enumerate(β1s)
        flag2 = flag
        ϵ1 = AS.risk_cc_convex_inv(β1, ζ1, samplesize)
        A1 = ϵ1 * nmode * sqrt(C1)
        if A1 < 1
            bound1 = 1 / AS.area_2sided_cap_inv(A1, nvar)
            bounds1[i] = min(bound1, def_bound1)
        else
            flag2 = false
        end
        if flag2
            prods[i] = bound1 * bound2
        end
    end
    push!(bounds, (samplesize, bounds1, bound2, prods))
end

@assert length(ratios) == nexp * nsize
@assert length(bounds) == nsize

# file = open(string(@__DIR__, "/output.txt"), "w")
# for row in eachrow(ratio_mat)
#     println(file, row)
# end
# close(file)

display(ratios)
display(bounds)

plt1 = plot(xlabel=L"N_1", ylabel="ratio")
plt2 = plot(xlabel=L"N_1", ylabel="bounds2")
boxplot!(plt1, getindex.(ratios, 1), getindex.(ratios, 2), label=nothing)
for (i, β1) in enumerate(β1s)
    iopt = argmin(j -> bounds[j][4][i], eachindex(bounds))
    plot!(plt1, getindex.(bounds, 1), getindex.(getindex.(bounds, 2), i), label="$(β1)")
    plot!(plt2, getindex.(bounds, 1), getindex.(getindex.(bounds, 4), i), label="$(β1)")
    scatter!(plt2, [bounds[iopt][1]], [bounds[iopt][4][i]], ms=5, label=nothing)
end
plot!(plt2, getindex.(bounds, 1), getindex.(bounds, 3), label="g")
display(plot(plt1, plt2))

end # module
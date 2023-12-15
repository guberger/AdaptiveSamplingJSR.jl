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

nvar = 2
nmode = 2
γ_min = 0
γ_max = 10

A_list = [randn(nvar, nvar) for _ = 1:nmode]

samplesize_min = 50
samplesize_max = 500
nsize = 2^3
samplesize_list_raw = range(samplesize_min, samplesize_max, length=nsize)
samplesize_list = ceil.(Int, samplesize_list_raw)
nexp = 2^2
ratios = Tuple{Int,Float64}[]
bounds = Tuple{Int,Vector{Float64},Vector{Float64}}[]
def_ratio = 20
def_bound = 20
βs = [0.05, 0.25, 0.5, 0.75]
ζ = nvar * (nvar + 1) / 2
C = 5

for samplesize in samplesize_list
    println("samplesize: ", samplesize, " ------------------------------------")
    bounds_cc = fill(float(def_bound), length(βs))
    bounds_pr = fill(float(def_bound), length(βs))
    for (i, β) in enumerate(βs)
        # Chance-constraint
        ϵ = AS.risk_cc_convex_inv(β, ζ, samplesize)
        area = ϵ * nmode * C^((nvar - 1) / 2)
        if area < 1
            s = AS.area_2sided_cap_inv(area, nvar)
            if s ≥ 1 / def_bound
                bounds_cc[i] = 1 / s
            end
        end
        # Projection
        ϵ = AS.risk_cc_convex_inv(β, 1, samplesize)
        area = ϵ * nmode * C^(nvar / 4)
        if area < 1
            s = AS.area_2sided_cap_inv(area, nvar)
            if s ≥ 1 / def_bound
                bounds_pr[i] = 1 / s
            end
        end
    end
    push!(bounds, (samplesize, bounds_cc, bounds_pr))
    for iexp in 1:nexp
        trans_list = AS.Transition[]
        for _ = 1:samplesize
            x = randn(nvar)
            q = rand(eachindex(A_list))
            y = A_list[q] * x
            push!(trans_list, AS.Transition(x, y))
        end
        γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, C,
                                            γ_min, γ_max, solverLP,
                                            out=false)
        if !isposdef(P)
            push!(ratios, (samplesize, def_ratio))
            continue
        end
        # singvals = svdvals(P)
        # κ = prod(singvals) / minimum(singvals)
        display(cond(P))
        γ_wb = AS.contraction_factor(A_list, P, solverSDP)
        ratio = γ_wb / γ
        push!(ratios, (samplesize, ratio))
    end
end

@assert length(ratios) == nexp * nsize
@assert length(bounds) == nsize

file = open(string(@__DIR__, "/output.txt"), "w")
for row in eachrow(ratios)
    println(file, row[1])
end
println(file, (1, eps()))
close(file)

display(ratios)
display(bounds)

plt = plot(xlabel=L"N", ylabel=L"\bar{r},\:\bar{f}",
           title=latexstring("n=$(nvar), m=$(nmode)"),
           ylim=(0.95,2.1), yscale=:log10)
boxplot!(plt, getindex.(ratios, 1), getindex.(ratios, 2), label=nothing)
ext_bound(i) = getindex.(bounds, i)
for (i, β) in enumerate(βs)
    plot!(plt, ext_bound(1), getindex.(ext_bound(2), i), label="CC: $(β)")
    plot!(plt, ext_bound(1), getindex.(ext_bound(3), i), ls=:dot, label="Heur: $(β)")
end
display(plt)
savefig(plt, "experiments/figures/onestep_$(nvar)_$(nmode)_$(nexp)_$(nsize).png")

end # module
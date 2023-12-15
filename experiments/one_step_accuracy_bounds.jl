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

SEED = 0
nvar = 3
nmode = 3
C = 5

samplesize_min = 50
samplesize_max = 500
nsize = 2^5
samplesize_list_raw = range(samplesize_min, samplesize_max, length=nsize)
samplesize_list = ceil.(Int, samplesize_list_raw)
nexp = 2^5

other_info = hash((SEED, C, samplesize_list...))
fname = "/results_one_step_$(nvar)_$(nmode)_$(nexp)_$(other_info).txt"
lines = readlines(string(@__DIR__, fname))
ratios = Tuple{Int,Float64}[]

ratio_found = false
for ln in lines
    if ln == "ratios:"
        global ratio_found = true
        continue
    end
    if !ratio_found
        continue
    end
    ln = replace(ln, "("=>"")
    ln = replace(ln, ")"=>"")
    words = split(ln, ", ")
    @assert length(words) == 2
    ratio = parse(Float64, words[2])
    ratio = min(ratio, 1e5)
    push!(ratios, (parse(Int, words[1]), ratio))
end

@assert length(ratios) == nsize * nexp

βs = [0.05, 0.25, 0.5]
ζ = nvar * (nvar + 1) / 2
bounds = Tuple{Int,Vector{Float64},Vector{Float64}}[]

for samplesize in samplesize_list
    bounds_cc = fill(1 / eps(), length(βs))
    bounds_he = fill(1 / eps(), length(βs))
    for (i, β) in enumerate(βs)
        # Chance-constraint
        ϵ = AS.risk_cc_convex_inv(β, ζ, samplesize)
        area = ϵ * nmode * C^((nvar - 1) / 2)
        if area < 1
            s = AS.area_2sided_cap_inv(area, nvar)
            if s > eps()
                bounds_cc[i] = 1 / s
            end
        end
        # Heuristic
        ϵ = AS.risk_cc_convex_inv(β, 1, samplesize - ζ)
        area = ϵ * nmode * C^(nvar / 4)
        if area < 1
            s = AS.area_2sided_cap_inv(area, nvar)
            if s > eps()
                bounds_he[i] = 1 / s
            end
        end
    end
    push!(bounds, (samplesize, bounds_cc, bounds_he))
end

@assert length(bounds) == nsize

display(ratios)
display(bounds)

plt = plot(xlabel=L"N", ylabel=L"\bar{r},\:\bar{f}",
           title=latexstring("n=$(nvar), m=$(nmode)"),
           ylim=(0.999,5), yscale=:log10)
boxplot!(plt, getindex.(ratios, 1), getindex.(ratios, 2), label=nothing)
ext_bound(i) = getindex.(bounds, i)
for (i, β) in enumerate(βs)
    plot!(plt, ext_bound(1), getindex.(ext_bound(2), i), lw=2, label=latexstring("\\beta=$(β)"))
end
display(plt)
savefig(plt, "experiments/figures/onestep_$(nvar)_$(nmode)_$(nexp)_$(nsize)_theory.png")

plt = plot(xlabel=L"N", ylabel=L"\bar{r},\:\bar{f}",
           title=latexstring("n=$(nvar), m=$(nmode)"),
           ylim=(0.999,2), yscale=:log10)
boxplot!(plt, getindex.(ratios, 1), getindex.(ratios, 2), label=nothing)
ext_bound(i) = getindex.(bounds, i)
for (i, β) in enumerate(βs)
    plot!(plt, ext_bound(1), getindex.(ext_bound(3), i), lw=2, label=latexstring("\\beta=$(β)"))
end
display(plt)
savefig(plt, "experiments/figures/onestep_$(nvar)_$(nmode)_$(nexp)_$(nsize)_heuristic.png")

end # module
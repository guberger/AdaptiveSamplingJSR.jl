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
Random.seed!(SEED)

nvar = 5
nmode = 5
C = 5
γ_min = 0
γ_max = 10

A_list = [randn(nvar, nvar) for _ = 1:nmode]

samplesize_min = 50
samplesize_max = 500
nsize = 2^5
samplesize_list_raw = range(samplesize_min, samplesize_max, length=nsize)
samplesize_list = ceil.(Int, samplesize_list_raw)
nexp = 2^5
ratios = Tuple{Int,Float64}[]

for samplesize in samplesize_list
    println("samplesize: ", samplesize, " ------------------------------------")
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
            push!(ratios, (samplesize, Inf))
            continue
        end
        display(cond(P))
        γ_wb = AS.contraction_factor(A_list, P, solverSDP)
        ratio = γ_wb / γ
        push!(ratios, (samplesize, ratio))
    end
end

@assert length(ratios) == nexp * nsize

other_info = hash((SEED, C, samplesize_list...))
fname = "/results_one_step_$(nvar)_$(nmode)_$(nexp)_$(other_info).txt"
file = open(string(@__DIR__, fname), "w")
println(file, "seed: $(SEED)")
println(file, "C: $(C)")
println(file, "samplesizes: $(samplesize_list)")
println(file, "ratios:")
for row in eachrow(ratios)
    println(file, row[1])
end
close(file)

end # module
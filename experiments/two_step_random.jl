module Experiment

using LinearAlgebra
using Random
using LaTeXStrings
using Plots
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

N = 4
M = 4
γ_min = 0
γ_max = 10

A_list = [randn(N, N) for _ = 1:M]

ND_max = 500
nD = 50
ND_list = ceil.(Int, range(10, ND_max, length=nD))
nexp = 30
ratio_list = fill(1e2, nD, nexp)

for (iD, ND) in enumerate(ND_list)
    for iexp in 1:nexp
        trans_list = AS.Transition[]
        for i = 1:ND
            x = randn(N)
            q = rand(eachindex(A_list))
            y = A_list[q] * x
            push!(trans_list, AS.Transition(x, y))
        end
        γ, P = AS.jsr_quadratic_data_driven(N, trans_list,
                                            γ_min, γ_max, solverLP,
                                            out=false)
        if !isposdef(P)
            continue
        end
        γ_wb = AS.contraction_factor(A_list, P, solverSDP)
        ratio_list[iD, iexp] = γ_wb / γ
    end
end

file = open(string(@__DIR__, "/output.txt"), "w")
for row in eachrow(ratio_list)
    println(file, row)
end
close(file)

display(A_list)

ratio_mean = sum(ratio_list, dims=2) / nexp
ratio_sorted = sort(ratio_list, dims=2)
jcol = floor(Int, nexp * 90 / 100)
ratio_90 = ratio_sorted[:, jcol]

plt = plot(xlabel=L"N_1", ylabel=L"\kappa(P_2)", yscale=:log10)
plot!(plt, ND_list, ratio_mean, marker=:d)
plot!(plt, ND_list, ratio_90, marker=:d)
display(plt)

end # module
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

samplesize_min = 50
samplesize_max = 500
nsize = 2^5
samplesize_list_raw = range(samplesize_min, samplesize_max, length=nsize + 2)
samplesize_list = ceil.(Int, samplesize_list_raw[2:(end - 1)])
nexp = 2^5
ratios = Tuple{Int,Float64}[]
rads = Tuple{Int,Float64}[]
bounds = Tuple{Int,Vector{Float64},Vector{Float64},Float64,Float64,Vector{Float64}}[]
def_ratio = 5
def_rad = 5
def_bound1 = 5
def_bound2 = 5
def_prod = 5
β1s = [0.05, 0.1, 0.25, 0.5]
β2 = 0.05
ζ1 = nvar * (nvar + 1) / 2
C1 = 5

for samplesize in samplesize_list
    println("samplesize: ", samplesize, " ------------------------------------")
    samplesize2 = samplesize_max - samplesize
    ϵ2 = AS.risk_cc_convex_inv(β2, 1, samplesize2)
    A2 = ϵ2 * nmode
    flag2 = false
    bound2 = def_bound2
    if A2 < 1
        s1 = AS.area_2sided_cap_inv(A2, nvar)
        if s1 ≥ 1 / def_bound2
            bound2 = 1 / s1
            flag2 = true
        end
    end
    bounds1_cc = fill(float(def_bound1), length(β1s))
    bounds1_proj = fill(float(def_bound1), length(β1s))
    prods = fill(float(def_prod), length(β1s))
    for (i, β1) in enumerate(β1s)
        ϵ1 = AS.risk_cc_convex_inv(β1, ζ1, samplesize)
        A1 = ϵ1 * nmode * sqrt(C1)
        if A1 < 1
            s1 = AS.area_2sided_cap_inv(A1, nvar)
            if s1 ≥ 1 / def_bound1
                bounds1_cc[i] = 1 / s1
            end
        end
        A1 = nmode * (1 - β1^(1 / samplesize)) * sqrt(C1)
        flag1 = false
        if A1 < 1
            s1 = AS.area_2sided_cap_inv(A1, nvar)
            if s1 ≥ 1 / def_bound1
                bounds1_proj[i] = 1 / s1
                flag1 = true
            end
        end
        if flag1 && flag2
            prods[i] = bounds1_proj[i] * bound2
        end
    end
    A1 = nmode / samplesize
    bound1_pack = def_bound1
    if A1 < 1
        s1 = AS.area_2sided_cap_inv(A1, nvar)
        den = (1 - C1 * (1 - s1))
        if den > 1 / def_bound1
            bound1_pack = 1 / den
        end
    end
    push!(bounds, (samplesize, bounds1_cc, bounds1_proj, bound1_pack, bound2, prods))
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
        if !isposdef(P)
            push!(ratios, (samplesize, def_ratio))
            push!(rads, (samplesize, def_rad))
            continue
        end
        # singvals = svdvals(P)
        # κ = prod(singvals) / minimum(singvals)
        display(cond(P))
        γ_wb = AS.contraction_factor(A_list, P, solverSDP)
        ratio = γ_wb / γ
        push!(ratios, (samplesize, ratio))
        if !flag2
            push!(rads, (samplesize, def_rad))
            continue
        end
        B = cholesky(P).U
        Bi = inv(B)
        γ2 = -1.0
        for _ = 1:samplesize2
            x = randn(nvar)
            q = rand(eachindex(A_list))
            y = B * (A_list[q] * (Bi * x))
            γ2 = max(γ2, norm(y) / norm(x))
        end
        rad = γ2 * bound2
        push!(rads, (samplesize, rad))
    end
end

@assert length(ratios) == nexp * nsize
@assert length(rads) == nexp * nsize
@assert length(bounds) == nsize

# file = open(string(@__DIR__, "/output.txt"), "w")
# for row in eachrow(ratio_mat)
#     println(file, row)
# end
# close(file)

display(ratios)
display(rads)
display(bounds)

plt1 = plot(xlabel=L"N_1", ylabel="ratio", ylim=(0.9, 2))
plt2 = plot(xlabel=L"N_1", ylabel="bounds", ylim=(0.9, 2))
plt3 = plot(xlabel=L"N_1", ylabel="rad", ylim=(2.05, 3.6))
boxplot!(plt1, getindex.(ratios, 1), getindex.(ratios, 2), label=nothing)
plot!(plt1, getindex.(bounds, 1), getindex.(bounds, 4), ls=:dash, label=nothing)
boxplot!(plt3, getindex.(rads, 1), getindex.(rads, 2), label=nothing)
for (i, β1) in enumerate(β1s)
    iopt = argmin(j -> bounds[j][6][i], eachindex(bounds))
    plot!(plt1, getindex.(bounds, 1), getindex.(getindex.(bounds, 2), i), label="$(β1)")
    plot!(plt1, getindex.(bounds, 1), getindex.(getindex.(bounds, 3), i), ls=:dot, label="$(β1)")
    plot!(plt2, getindex.(bounds, 1), getindex.(getindex.(bounds, 6), i), label="$(β1)")
    scatter!(plt2, [bounds[iopt][1]], [bounds[iopt][6][i]], ms=5, label=nothing)
end
plot!(plt2, getindex.(bounds, 1), getindex.(bounds, 5), label="g")
display(plot(plt1, plt2, plt3))

end # module
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

A_list = [
    [1 1; 0 1],
    [1 -1; 0 1],
]

α1 = π / 3.1
α2 = 2.1 * π / 3.1
r = 3
A_list = [
    [cos(α1) (-r * sin(α1)); (sin(α1) / r) cos(α1)],
    [cos(α2) (-r * sin(α2)); (sin(α2) / r) cos(α2)],
]
A_list = [round.(A, digits=2) for A in A_list]
display(A_list)

samplesize = 25
β = 0.05
ζ = nvar * (nvar + 1) / 2
C = 10

np = 100
circle = [[cos(α), sin(α)] for α in range(0, 2π, length=np)]
plt = plot(xlabel=L"x[1]", ylabel=L"x[2]", aspect_ratio=:equal)
plot!(plt, getindex.(circle, 1), getindex.(circle, 2),
      c=:gray, ls=:dash, label=false)

γ, P = AS.jsr_quadratic_white_box(nvar, A_list, C,
                                  γ_min, γ_max, solverSDP,
                                  out=false, ϵ=1e-3)
@assert isposdef(P)
singvals = svdvals(P)
κ = sqrt(prod(singvals) / minimum(singvals)^nvar)
display(κ)

Binv = cholesky(P).U
B = inv(Binv)
ellipse = [B * x for x in circle]

plot!(plt, getindex.(ellipse, 1), getindex.(ellipse, 2), label=L"N=\infty")

#-------------------------------------------------------------------------------

trans_list = AS.Transition[]
for _ = 1:(2 * samplesize)
    x = randn(nvar)
    q = rand(eachindex(A_list))
    y = A_list[q] * x
    push!(trans_list, AS.Transition(x, y))
end

γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, C,
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
display(γ * factor)
display(1)

Binv = cholesky(P).U
B = inv(Binv)

ellipse = [B * x for x in circle]
initials = [normalize(trans.x) for trans in trans_list]

plot!(plt, getindex.(ellipse, 1), getindex.(ellipse, 2),
      label=latexstring("N=$(2 * samplesize)"))
scatter!(plt, getindex.(initials, 1), getindex.(initials, 2),
         ms=3, label=false)

#-------------------------------------------------------------------------------

@assert length(trans_list) == 2 * samplesize
trans_list = trans_list[1:samplesize]

γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, C,
                                    γ_min, γ_max, solverLP,
                                    out=false)
@assert isposdef(P)
singvals = svdvals(P)
κ = sqrt(prod(singvals) / minimum(singvals)^nvar)
display(κ)
Binv = cholesky(P).U
B = inv(Binv)

ellipse = [B * x for x in circle]
initials = [normalize(trans.x) for trans in trans_list]

plot!(plt, getindex.(ellipse, 1), getindex.(ellipse, 2),
      label=latexstring("N=$(samplesize)"))
scatter!(plt, getindex.(initials, 1), getindex.(initials, 2),
         ms=3, label=false)

empty!(trans_list)
for _ = 1:samplesize
    x = randn(nvar)
    xp = B * x
    q = rand(eachindex(A_list))
    yp = A_list[q] * xp
    y = Binv * yp
    push!(trans_list, AS.Transition(x, y))
end

γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, 2,
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
display(γ * factor)

initials = [normalize(trans.x) for trans in trans_list]
initials_ellipse = [B * x for x in initials]

scatter!(plt, getindex.(initials_ellipse, 1), getindex.(initials_ellipse, 2),
         ms=3, label=false)

savefig(plt, "experiments/figures/example_2_2_ill_sampling.png")
display(plt)

#-------------------------------------------------------------------------------

samplesize1_list = 4:45
samplesize_total = 50
expected_ratio = Tuple{Int,Float64}[]

for samplesize1 in samplesize1_list
    samplesize2 = samplesize_total - samplesize1
    ϵ = AS.risk_cc_convex_inv(0.5, 1, samplesize1 - ζ)
    area = ϵ * nmode * C^(nvar / 4)
    s = AS.area_2sided_cap_inv(area, nvar)
    factor1 = 1 / s
    ϵ = AS.risk_cc_convex_inv(β, ζ, samplesize2)
    area = ϵ * nmode
    s = AS.area_2sided_cap_inv(area, nvar)
    factor2 = 1 / s
    push!(expected_ratio, (samplesize1, factor1 * factor2))
end

the_min = argmin(t -> t[2], expected_ratio)
display(the_min)
display(expected_ratio)

plt = plot(xlabel=L"N_1", ylabel=L"\bar{f}")
plot!(plt, getindex.(expected_ratio, 1), getindex.(expected_ratio, 2))
scatter!(plt, [the_min[1]], [the_min[2]])
savefig(plt, "experiments/figures/example_2_2_ill_heuristic_N1.png")
display(plt)

#-------------------------------------------------------------------------------

bounds = Tuple{Int,Float64}[]

for samplesize1 in samplesize1_list
    samplesize2 = samplesize_total - samplesize1
    
    empty!(trans_list)
    for _ = 1:samplesize1
        x = randn(nvar)
        q = rand(eachindex(A_list))
        y = A_list[q] * x
        push!(trans_list, AS.Transition(x, y))
    end
    
    @assert length(trans_list) == samplesize1
    γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, C,
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
    γ, P = AS.jsr_quadratic_data_driven(nvar, trans_list, 2,
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
savefig(plt, "experiments/figures/example_2_2_ill_different_N1.png")
display(plt)

end # module
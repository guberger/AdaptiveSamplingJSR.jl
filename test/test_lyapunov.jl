module TestSet

using Test
using LinearAlgebra
using JuMP
using HiGHS
using CSDP

@static if isdefined(Main, :TestLocal)
    include("../src/AdaptiveSamplingJSR.jl")
else
    using AdaptiveSamplingJSR
end
const AS = AdaptiveSamplingJSR

solverLP() = Model(optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent()=>true))
solverSDP() = Model(optimizer_with_attributes(CSDP.Optimizer, MOI.Silent()=>true))

using CSDP
solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
C = 10

@testset "JSR normal" begin
    D = [5.0, 3.0]
    θ1 = 1.0
    A1 = D .* [cos(θ1) sin(θ1); -sin(θ1) cos(θ1)] ./ D'
    θ2 = 2.0
    A2 = D .* [cos(θ2) sin(θ2); -sin(θ2) cos(θ2)] ./ D'
    α_list = range(0, 2π, length=101)[1:100]
    trans_list = AS.Transition[]
    for α in α_list
        x = [sin(α), cos(α)]
        for A in (A1, A2)
            y = A * x
            push!(trans_list, AS.Transition(x, y))
        end
    end
    γ, P = AS.jsr_quadratic_data_driven(2, trans_list, C, 0, 10, solverLP)
    @test 0.999 < γ < 1.001
    @test cond(D .* P .* D') < 1.001
    γ, P = AS.jsr_quadratic_data_driven(2, trans_list, 1, 0, 10, solverLP)
    @test abs(γ - maximum(A -> opnorm(A), (A1, A2))) < 1e-4
    @test cond(P) ≈ 1
    γ, P = AS.jsr_quadratic_white_box(2, (A1, A2), C, 0, 10, solverSDP, ϵ=0.25)
    @test 0.9 < γ < 1.1
    @test cond(D .* P .* D') < 1.5
    γ, P = AS.jsr_quadratic_white_box(2, (A1, A2), 3, 0, 10, solverSDP, ϵ=0.25)
    @test cond(P) < 3.0001
    P = diagm(1 ./ D.^2)
    γ = AS.contraction_factor((A1, A2), P, solverSDP)
    @test 0.999 < γ < 1.001
end

@testset "JSR degenerate" begin
    A1 = [1 0; 1 0] / sqrt(2)
    A2 = [0 1; 0 -1] / sqrt(2)
    α_list = range(0, 2π, length=101)[1:100]
    trans_list = AS.Transition[]
    for α in α_list
        x = [sin(α), cos(α)]
        for A in (A1, A2)
            y = A * x
            push!(trans_list, AS.Transition(x, y))
        end
    end
    γ, P = AS.jsr_quadratic_data_driven(2, trans_list, C, 0, 10, solverLP)
    @test 0.999 < γ < 1.001
    @test cond(P) < 1.001
    γ, P = AS.jsr_quadratic_white_box(2, (A1, A2), C, 0, 10, solverSDP, ϵ=0.25)
    @test 0.9 < γ < 1.1
    @test cond(P) < 1.5
    P = Matrix{Float64}(I, 2, 2)
    γ = AS.contraction_factor((A1, A2), P, solverSDP)
    @test 0.999 < γ < 1.001
end

@testset "JSR degenerate" begin
    A1 = [0 -1; 1 0]
    P = [4 0; 0 1]
    γ = AS.contraction_factor((A1,), P, solverSDP)
    @test 1.999 < γ < 2.001
end

end # module
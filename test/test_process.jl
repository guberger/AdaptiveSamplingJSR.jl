module TestSet

using Test
using LinearAlgebra
using JuMP
using HiGHS

@static if isdefined(Main, :TestLocal)
    include("../src/AdaptiveSamplingJSR.jl")
else
    using AdaptiveSamplingJSR
end
const AS = AdaptiveSamplingJSR

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

@testset "Process" begin
    @test AS.adaptive_sampling_process(5, 0, 0, 0)
end

end # module
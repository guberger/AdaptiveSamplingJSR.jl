module TestSet

using Test

@static if isdefined(Main, :TestLocal)
    include("../src/AdaptiveSamplingJSR.jl")
else
    using AdaptiveSamplingJSR
end
const AS = AdaptiveSamplingJSR

f(i, ϵ, N) = binomial(N, i) * ϵ^i * (1 - ϵ)^(N - i)

@testset "Risk" begin
    for ϵ in range(0.01, 0.99, length=5)
        for N = 1:6
            for ζ = 1:N
                β = AS.risk_cc_convex(ϵ, ζ, N)
                @test β ≈ sum(i -> f(i, ϵ, N), 0:ζ-1)
                @test abs(ϵ - AS.risk_cc_convex_inv(β, ζ, N)) < 1e-6
            end
        end
    end
end

end # module
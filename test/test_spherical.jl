module TestSet

using Test

@static if isdefined(Main, :TestLocal)
    include("../src/AdaptiveSamplingJSR.jl")
else
    using AdaptiveSamplingJSR
end
const AS = AdaptiveSamplingJSR

@testset "Spherical" begin
    for s in range(0, 0.99, length=5)
        @test AS.area_2sided_cap(s, 2) ≈ 2 * acos(s) / π
        @test AS.area_2sided_cap(s, 3) ≈ 1 - s
    end
    for A in range(0, 0.99, length=5)
        @test AS.area_2sided_cap_inv(A, 2) ≈ cos(π * A / 2)
        @test AS.area_2sided_cap_inv(A, 3) ≈ 1 - A
    end
end

end # module
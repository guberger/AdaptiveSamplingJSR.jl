module TestSet

using Test
@static if isdefined(Main, :TestLocal)
    include("../src/AdaptiveSamplingJSR.jl")
else
    using AdaptiveSamplingJSR
end

@testset "Greetings" begin
    stdout_old = Base.stdout
    Base.stdout = IOBuffer()
    AdaptiveSamplingJSR.greet()
    @test String(take!(stdout)) == "Hello World! I am AdaptiveSamplingJSR!"
    Base.stdout = stdout_old
end

end # module
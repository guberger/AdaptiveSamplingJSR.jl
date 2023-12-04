function sample_point(N)
    x = randn(N)
    return x / norm(x)
end

function sample_mode(M)
    return rand(1:M)
end

function adaptive_sampling_process(N, A_list, γ_min, γ_max;
                                   max_iter=Inf, δ=1e-5, ϵ=1e-5)
    iter = 0
    P_opt = Matrix{Float64}(I, N, N)
    γ_opt = Inf
    B = cholesky(P_opt)
    while iter < max_iter
        iter += 1
        println("hey")
        if iter > 5
            break
        end
    end
    return true    
end
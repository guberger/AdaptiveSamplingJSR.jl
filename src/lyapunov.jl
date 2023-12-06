struct Transition
    x::Vector{Float64}
    y::Vector{Float64}
end

function jsr_quadratic_white_box(N, A_list, C,
                                 γ_min, γ_max, solver;
                                 max_val=1e5, ϵ=1e-3, out=true)
    return jsr_quadratic(N, (), A_list, C, γ_min, γ_max, solver,
                         max_val=max_val, ϵ=ϵ, out=out)
end

function jsr_quadratic_data_driven(N, trans_list, C,
                                   γ_min, γ_max, solver;
                                   max_val=1e5, ϵ=1e-5, out=true, sdp=false)
    return jsr_quadratic(N, trans_list, (), C, γ_min, γ_max, solver,
                         max_val=max_val, ϵ=ϵ, out=out, sdp=sdp)
end

function jsr_quadratic(N, trans_list, A_list, C,
                       γ_min, γ_max, solver;
                       max_val=1e5, ϵ=1e-3, out=true, sdp=true)
    γ_up = γ_max
    γ_lo = γ_min
    γ_opt = 0.0
    P_opt = Matrix{Float64}(I, N, N)
    @assert C ≥ 1

    while γ_up - γ_lo > ϵ
        γ = (γ_up + γ_lo)/2
        out && @printf("%f - [%f, %f]: ", γ, γ_lo, γ_up)
        P, flag = stability_quadratic(N, trans_list, A_list, C,
                                      γ, max_val, solver, sdp=sdp)

        if flag
            out && println("ok")
            γ_up = γ
            copy!(P_opt, P)
            γ_opt = γ
        else
            out && println("ko")
            γ_lo = γ
        end
    end

    @printf("Interval for γ: %f - %f\n", γ_lo, γ_up)
    return γ_opt, P_opt
end

function stability_quadratic(N, trans_list, A_list, C,
                             γ, max_val, solver; sdp=true)
    γ2 = γ^2
    model = solver()
    P = @variable(model, [1:N, 1:N], Symmetric,
                  lower_bound=-max_val, upper_bound=max_val)

    if sdp
        @constraint(model, P - I in PSDCone())
        @constraint(model, C*I - P in PSDCone())
    end    

    for trans in trans_list
        x, y = trans.x, trans.y
        px = x' * P * x
        py = y' * P * y
        if !sdp
            @constraint(model, px ≥ norm(x)^2)
            @constraint(model, py ≥ norm(y)^2)
            @constraint(model, px ≤ C * norm(x)^2)
            @constraint(model, py ≤ C * norm(y)^2)
        end
        @constraint(model, py ≤ γ2 * px)
    end

    for A in A_list
        @constraint(model, γ2 * P - A' * P * A in PSDCone())
    end

    @objective(model, Min, tr(P))

    optimize!(model)

    if primal_status(model) == FEASIBLE_POINT
        return value.(P), true
    elseif termination_status(model) == INFEASIBLE
        return Matrix{Float64}(I, N, N), false
    else
        println(solution_summary(model))
        error("jsr")
    end
end

function contraction_factor(A_list, P, solver)
    model = solver()
    γ2 = @variable(model, lower_bound=-1)
  
    for A in A_list
        @constraint(model, γ2 * P - A' * P * A in PSDCone())
    end

    @objective(model, Min, γ2)

    optimize!(model)

    if primal_status(model) == FEASIBLE_POINT
        return sqrt(value(γ2))
    else
        println(solution_summary(model))
        error("contraction")
    end
end
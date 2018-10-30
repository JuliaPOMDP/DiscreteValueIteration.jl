struct SparseValueIterationSolver <: Solver
    max_iterations::Int64
    belres::Float64 # the Bellman Residual
    include_Q::Bool
    verbose::Bool
    init_util::Vector{Float64}
end

function SparseValueIterationSolver(;max_iterations=500,
                                    belres::Float64=1e-3,
                                    include_Q::Bool=true,
                                    verbose::Bool=false,
                                    init_util::Vector{Float64}=Vector{Float64}(undef, 0))
    return SparseValueIterationSolver(max_iterations, belres, include_Q, verbose, init_util)
end

@POMDP_require solve(solver::SparseValueIterationSolver, mdp::Union{MDP,POMDP}) begin
    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @req n_states(::P)
    @req n_actions(::P)
    @subreq ordered_states(mdp)
    @subreq ordered_actions(mdp)
    @req transition(::P,::S,::A)
    @req reward(::P,::S,::A)
    @req stateindex(::P,::S)
    @req actionindex(::P, ::A)
    @req actions(::P, ::S)
    as = actions(mdp)
    ss = states(mdp)
    a = first(as)
    s = first(ss)
    dist = transition(mdp, s, a)
    D = typeof(dist)
    @req support(::D)
    @req pdf(::D,::S)
end

function qvalue!(m::Union{MDP,POMDP}, transition_A_S_S2, reward_S_A::AbstractMatrix{F}, value_S::AbstractVector{F}, out_qvals_S_A) where {F}
    @assert size(out_qvals_S_A) == (n_states(m), n_actions(m))
    for a in 1:n_actions(m)
        out_qvals_S_A[:, a] = view(reward_S_A, :, a) + discount(m) * transition_A_S_S2[a] * value_S
    end
end

function transition_matrix_a_s_sp(mdp::MDP)
    # Thanks to zach
    na = n_actions(mdp)
    ns = n_states(mdp)
    transmat_row_A = [Float64[] for _ in 1:n_actions(mdp)]
    transmat_col_A = [Float64[] for _ in 1:n_actions(mdp)]
    transmat_data_A = [Float64[] for _ in 1:n_actions(mdp)]

    for a in actions(mdp)
        ai = actionindex(mdp, a)
        for s in states(mdp)
            si = stateindex(mdp, s)
            if !isterminal(mdp, s) # if terminal, the transition probabilities are all just zero
                td = transition(mdp, s, a)
                for (sp, p) in weighted_iterator(td)
                    if p > 0.0
                        spi = stateindex(mdp, sp)
                        push!(transmat_row_A[ai], si)
                        push!(transmat_col_A[ai], spi)
                        push!(transmat_data_A[ai], p)
                    end
                end
            end
        end
    end
    transmats_A_S_S2 = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], n_states(mdp), n_states(mdp)) for a in 1:n_actions(mdp)]
    # Note: not valid for terminal states
    # @assert all(all(sum(transmats_A_S_S2[a], dims=2) .≈ ones(n_states(mdp))) for a in 1:n_actions(mdp)) "Transition probabilities must sum to 1"
    return transmats_A_S_S2
end

function reward_s_a(mdp::MDP)
    reward_S_A = zeros(n_states(mdp), n_actions(mdp))
    for s in states(mdp)
        for a in actions(mdp)
            reward_S_A[stateindex(mdp, s), actionindex(mdp, a)] = reward(mdp, s, a)
        end
    end
    return reward_S_A
end

function solve(solver::SparseValueIterationSolver, mdp::Union{MDP, POMDP})
    nS = n_states(mdp)
    nA = n_actions(mdp)
    if isempty(solver.init_util)
        v_S = zeros(nS)
    else
        @assert length(solver.init_util) == nS "Input utility dimension mismatch"
        v_S = solver.init_util
    end

    transition_A_S_S2 = transition_matrix_a_s_sp(mdp)
    reward_S_A = reward_s_a(mdp)
    qvals_S_A = zeros(nS, nA)
    maxchanges_T = zeros(solver.max_iterations)

    total_time = 0.0
    for i in 1:solver.max_iterations
        iter_time = @elapsed begin
            qvalue!(mdp, transition_A_S_S2, reward_S_A, v_S, qvals_S_A)
            new_v_S = dropdims(maximum(qvals_S_A, dims=2), dims=2)
            @assert size(v_S) == size(new_v_S)
            maxchanges_T[i] = maximum(abs.(new_v_S .- v_S))
            v_S = new_v_S
        end
        total_time += iter_time
        if solver.verbose
            @info "residual: $(maxchanges_T[i]), time: $(iter_time), total time: $(total_time) " i
        end
        maxchanges_T[i] < solver.belres ? break : nothing
    end
    qvalue!(mdp, transition_A_S_S2, reward_S_A, v_S, qvals_S_A)
    # Rounding to avoid floating point error noise
    policy_S = dropdims(getindex.(argmax(round.(qvals_S_A, digits=20), dims=2), 2), dims=2)
    if solver.include_Q
        policy = ValueIterationPolicy(mdp, qvals_S_A, v_S, policy_S)
    else
        policy = ValueIterationPolicy(mdp, utility=v_S, policy=policy_S, include_Q=false)
    end
    return policy
end

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

@POMDP_require solve(solver::SparseValueIterationSolver, mdp::MDP) begin
    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @req n_states(::P)
    @req n_actions(::P)
    @subreq ordered_states(mdp)
    @subreq ordered_actions(mdp)
    @req transition(::P,::S,::A)
    @req reward(::P,::S,::A,::S)
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
    @subreq SparseTabularMDP(mdp)
end

function qvalue!(m::Union{MDP,POMDP}, transition_A_S_S2, reward_S_A::AbstractMatrix{F}, value_S::AbstractVector{F}, out_qvals_S_A) where {F}
    @assert size(out_qvals_S_A) == (n_states(m), n_actions(m))
    for a in 1:n_actions(m)
        out_qvals_S_A[:, a] = view(reward_S_A, :, a) + discount(m) * transition_A_S_S2[a] * value_S
    end
end

function solve(solver::SparseValueIterationSolver, mdp::SparseTabularMDP)
    nS = n_states(mdp)
    nA = n_actions(mdp)
    if isempty(solver.init_util)
        v_S = zeros(nS)
    else
        @assert length(solver.init_util) == nS "Input utility dimension mismatch"
        v_S = solver.init_util
    end

    transition_A_S_S2 = transition_matrices(mdp)
    reward_S_A = reward_matrix(mdp)
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

function solve(solver::SparseValueIterationSolver, mdp::MDP)
    return solve(solver, SparseTabularMDP(mdp))
end
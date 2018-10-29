struct SparseValueIterationSolver <: Solver
    max_iterations::Int64
    belres::Float64 # the Bellman Residual
    verbose::Bool
    init_util::Vector{Float64}
end

function SparseValueIterationSolver(;max_iterations=500,
                                    belres::Float64 = 1e-3,
                                    verbose::Bool=false,
                                    init_util::Vector{Float64}=Vector{Float64}(undef, 0))
    return SparseValueIterationSolver(max_iterations, belres, verbose, init_util)
end

struct SparseVIPolicy{F}
    v_S::AbstractVector{F}
    qvals_S_A::AbstractArray{F, 2}
    policy_S::AbstractVector{Int}
    action_map::Vector # Maps the action index to the concrete action type
    mdp::Union{MDP,POMDP} # uses the model for indexing in the action function
end

function SparseVIPolicy(mdp::Union{MDP, POMDP}; utility::AbstractVector{Float64}=zeros(n_states(mdp)),
                        policy::AbstractVector{Int64}=zeros(Int64, n_states(mdp)))
    ns = n_states(mdp)
    na = n_actions(mdp)
    @assert length(utility) == ns "Input utility dimension mismatch"
    @assert length(policy) == ns "Input policy dimension mismatch"
    action_map = ordered_actions(mdp)
    qmat = zeros(ns, na)
    return SparseVIPolicy(utility, qmat, policy, action_map, mdp)
end

function SparseVIPolicy(mdp::Union{MDP, POMDP}, q::AbstractMatrix{Float64}, util::AbstractVector{Float64}, policy::AbstractVector{Int64})
    action_map = ordered_actions(mdp)
    return SparseVIPolicy(util, q, policy, action_map, mdp)
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
end

function qvalue!(m::Union{MDP,POMDP}, transition_A_S_S2, reward_S::AbstractVector{F}, value_S::AbstractVector{F}, out_qvals_S_A) where {F}
    @assert size(out_qvals_S_A) == (n_states(m), n_actions(m))
    for a in 1:n_actions(m)
        out_qvals_S_A[:, a] = reward_S + discount(m) * transition_A_S_S2[a] * value_S
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
            else
                push!(transmat_row_A[ai], si)
                push!(transmat_col_A[ai], si)
                push!(transmat_data_A[ai], 1.0)
            end
        end
    end
    transmats_A_S_S2 = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], n_states(mdp), n_states(mdp)) for a in 1:n_actions(mdp)]
    @assert all(all(sum(transmats_A_S_S2[a], dims=2) .≈ ones(n_states(mdp))) for a in 1:n_actions(mdp)) "Transition probabilities must sum to 1"
    return transmats_A_S_S2
end

function reward_s(mdp::MDP)
    reward_S = zeros(n_states(mdp))
    for s in states(mdp)
        reward_S[stateindex(mdp, s)] = reward(mdp, s)
    end
    return reward_S
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
    reward_S = reward_s(mdp)
    qvals_S_A = zeros(nS, nA)
    maxchanges_T = zeros(solver.max_iterations)

    total_time = 0.0
    for i in 1:solver.max_iterations
        iter_time = @elapsed begin
            qvalue!(mdp, transition_A_S_S2, reward_S, v_S, qvals_S_A)
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
    qvalue!(mdp, transition_A_S_S2, reward_S, v_S, qvals_S_A)
    # Rounding to avoid floating point error noise
    policy_S = dropdims(getindex.(argmax(round.(qvals_S_A, digits=20), dims=2), 2), dims=2)
    policy = SparseVIPolicy(mdp, qvals_S_A, v_S, policy_S)
end

function action(policy::SparseVIPolicy, s::S) where S
    sidx = stateindex(policy.mdp, s)
    aidx = policy.policy_S[sidx]
    return policy.action_map[aidx]
end

function value(policy::SparseVIPolicy, s::S) where S
    sidx = stateindex(policy.mdp, s)
    return policy.v_S[sidx]
end
